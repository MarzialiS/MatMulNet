#!/usr/bin/env python3
"""Projected pseudo-arclength continuation of a length-r candidate towards M_n.

    python continuation.py --self-test
    python continuation.py rank22_input.csv --out data

Input: a labelled CSV (term, F/H/K, n^2 real coordinates), or an NPZ with H, K of shape
(n^2, r) and F of shape (r, n^2). This is a local projected continuation, not a certified
solver of all tensor equations.

Let x0 be the input parameters, f(x) the n^6 tensor coefficients, and M matrix multiplication
in row-major coordinates. Define b(t) = M + t (f(x0) - M). At x0, compute J0 = L diag(s) V^T.
If q = rank(J0), set B = V[:, :q] and x(y) = x0 + B y, which imposes the fixed orthogonal slice
V[:, q:]^T (x - x0) = 0. Use a fixed Gaussian matrix R of size (q, n^6), seed 12345 by default,
and orthonormalise its row space: P = qr(R.T).Q.T. The tracked system is
g(y, t) = P (f(x(y)) - b(t)) / ||M|| = 0.

A Euclidean unit tangent is obtained by bordered linear solves. Each predictor
zpred = z + ds * tau is corrected with Newton's method on (g(z), tau.T (z - zpred)) = 0.
All n^6 residuals are monitored independently. The fixed projection, slice basis, starting
factors, input hash, numerical options, accepted steps, and best/last factors are saved.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import warnings
import numpy as np
import scipy
from scipy import linalg as la
from threadpoolctl import threadpool_limits


class TensorMap:
    def __init__(self, m: int, r: int):
        self.m, self.r = m, r
        self.eye = np.eye(m)
        self.size = 3*r*m

    def unpack(self, x):
        return x.reshape(3, self.r, self.m)

    def tensor(self, x):
        u,v,w = self.unpack(x)
        return np.einsum('sa,sb,sc->abc',u,v,w, optimize=True).ravel()

    def jacobian(self, x):
        u,v,w = self.unpack(x)
        m,r = self.m,self.r
        return np.concatenate([
            np.einsum('ai,sb,sc->abcsi',self.eye,v,w).reshape(m**3,r*m),
            np.einsum('sa,bi,sc->abcsi',u,self.eye,w).reshape(m**3,r*m),
            np.einsum('sa,sb,ci->abcsi',u,v,self.eye).reshape(m**3,r*m)
        ],axis=1)


def matmul_tensor(n):
    target = np.zeros((n*n,n*n,n*n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                target[n*i+j,n*j+k,n*i+k] = 1.
    return target.ravel()


def read_factors(path):
    if path.suffix.lower() == '.npz':
        with np.load(path,allow_pickle=False) as z:
            H,K,F = [np.asarray(z[k],dtype=float) for k in ('H','K','F')]
        if H.ndim != 2 or K.shape != H.shape or F.shape != H.T.shape:
            raise ValueError('Expected H,K shape (m,r), F shape (r,m).')
        a = np.stack((H.T,K.T,F))
    elif path.suffix.lower() == '.csv':
        factors = {k:{} for k in ('H','K','F')}
        with path.open(newline='',encoding='utf-8-sig') as fh:
            for lineno,row in enumerate(csv.reader(fh),1):
                if not row or not any(s.strip() for s in row):
                    continue
                if row[0].strip() == 'rank_1_tensor':
                    continue  # header row, wherever it appears
                if len(row)<3 or row[1].strip() not in factors:
                    raise ValueError(f'Unrecognised CSV row {lineno}; refusing to guess layout.')
                term, name = int(row[0]),row[1].strip()
                if term in factors[name]:
                    raise ValueError(f'Duplicate term {term} for {name}.')
                factors[name][term]=np.array(row[2:],dtype=float)
        r=len(factors['H']); expected=set(range(1,r+1))
        if r<1 or any(set(factors[k])!=expected for k in factors):
            raise ValueError('Each factor must contain exactly terms 1,...,r.')
        a=np.stack([np.stack([factors[k][s] for s in range(1,r+1)])
                    for k in ('H','K','F')])
    else:
        raise ValueError('Use the labelled CSV or an NPZ with H,K,F.')
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite factor entries.')
    m=a.shape[2]; n=int(round(np.sqrt(m)))
    if n*n!=m:
        raise ValueError('Each factor vector must have n^2 coordinates.')
    return a.ravel(),n,a.shape[1]


class ProjectedSystem:
    def __init__(self, model,x0,target,seed=12345,rcond=1e-10):
        self.model=model; self.x0=x0.copy(); self.target=target.copy()
        self.f0=model.tensor(x0); self.direction=self.f0-target
        self.target_norm=float(la.norm(target))
        if self.target_norm==0:
            raise ValueError('Zero target not supported.')
        _,s,vt=la.svd(model.jacobian(x0),full_matrices=False,check_finite=True)
        self.singular_values=s
        self.q=int(np.count_nonzero(s>rcond*s[0]))
        self.B=vt[:self.q].T.copy()
        self.S=vt[self.q:].copy()
        rng=np.random.default_rng(seed)
        R=rng.normal(size=(self.q,target.size))
        self.P=la.qr(R.T,mode='economic',check_finite=True)[0].T.copy()
        self.gt=-(self.P@self.direction)/self.target_norm

    def parameters(self,z):
        return self.x0+self.B@z[:-1]

    def residual(self,z):
        x=self.parameters(z)
        return self.P@(self.model.tensor(x)-self.target-z[-1]*self.direction)/self.target_norm

    def jacobian(self,z):
        x=self.parameters(z)
        jy=(self.P@(self.model.jacobian(x)@self.B))/self.target_norm
        return np.column_stack((jy,self.gt))

    def tangent(self,z,previous=None):
        J=self.jacobian(z)
        if previous is None:
            tau=np.r_[la.solve(J[:,:-1],J[:,-1],assume_a='gen'),-1.]
        else:
            A=np.vstack((J,previous))
            rhs=np.zeros(self.q+1);rhs[-1]=1.
            tau=la.solve(A,rhs,assume_a='gen')
            if tau@previous<0: tau=-tau
        return tau/la.norm(tau)

    def correct(self,zpred,tau,tol,maxiter):
        z=zpred.copy()
        for iteration in range(maxiter+1):
            g=np.r_[self.residual(z), tau@(z-zpred)]
            ng=float(la.norm(g))
            if not np.isfinite(ng): return None,iteration,float('inf')
            if ng<=tol: return z,iteration,ng
            if iteration==maxiter: break
            A=np.vstack((self.jacobian(z),tau))
            try:
                dz=la.solve(A,-g,assume_a='gen')
            except (la.LinAlgError,la.LinAlgWarning):
                return None,iteration,ng
            alpha=1.
            for _ in range(12):
                trial=z+alpha*dz
                trialnorm=la.norm(np.r_[self.residual(trial),tau@(trial-zpred)])
                if np.isfinite(trialnorm) and (trialnorm<ng*(1.-1e-4*alpha) or trialnorm<=tol):
                    z=trial
                    break
                alpha*=.5
            else: return None,iteration,ng
        return None,maxiter,ng

    def diagnostics(self,z,step,ds,newton_iters,tau,corrector_norm):
        x=self.parameters(z); value=self.model.tensor(x)
        u,v,w=self.model.unpack(x)
        weights=np.prod([la.norm(f,axis=1) for f in (u,v,w)],axis=0)
        b=self.target+z[-1]*self.direction
        absolute=float(la.norm(value-self.target))
        current=float(la.norm(value-b))
        return dict(step=int(step),t=float(z[-1]),ds=float(ds),
                    tangent_t=float(tau[-1]),newton_iters=int(newton_iters),
                    corrector_residual_normalized=float(corrector_norm),
                    projected_residual_normalized=float(la.norm(self.residual(z))),
                    projected_residual_absolute=float(la.norm(self.residual(z))*self.target_norm),
                    slice_residual=float(la.norm(self.S@(x-self.x0))),
                    full_residual_to_current_target=current,
                    relative_full_residual_to_current_target=current/float(la.norm(b)),
                    full_residual_to_exact=absolute,
                    relative_full_residual_to_exact=absolute/self.target_norm,
                    full_residual_to_start=float(la.norm(value-self.f0)),
                    param_norm=float(la.norm(x)),weight=float(weights.sum()),
                    max_term_scale=float(weights.max()),
                    max_u_norm=float(la.norm(u,axis=1).max()),
                    max_v_norm=float(la.norm(v,axis=1).max()),
                    max_w_norm=float(la.norm(w,axis=1).max()))

    def save_point(self,path,z):
        u,v,w=self.model.unpack(self.parameters(z))
        np.savez_compressed(path,H=u.T,K=v.T,F=w,t=z[-1],y=z[:-1],
                            target=self.target,current_target=self.target+z[-1]*self.direction)


def self_test():
    rng=np.random.default_rng(913)
    model=TensorMap(4,1);x0=rng.normal(size=model.size)
    d=rng.normal(size=model.size);d/=la.norm(d);h=1e-5
    analytic=model.jacobian(x0)@d
    fd=(model.tensor(x0+h*d)-model.tensor(x0-h*d))/(2*h)
    derivative_error=float(la.norm(analytic-fd)/la.norm(analytic))
    assert derivative_error<1e-8,derivative_error
    target=.9*model.tensor(x0)
    sys0=ProjectedSystem(model,x0,target)
    z=np.r_[np.zeros(sys0.q),1.];tau=sys0.tangent(z)
    for _ in range(20):
        pred=z+.01*tau
        nxt,it,err=sys0.correct(pred,tau,1e-12,15)
        assert nxt is not None
        z=nxt;tau=sys0.tangent(z,tau)
    full_relative=float(la.norm(model.tensor(sys0.parameters(z))-(target+z[-1]*sys0.direction))/la.norm(target))
    assert full_relative<1e-9,full_relative
    # Direct check of row-major indexing on the classical 2x2 tensor.
    M=matmul_tensor(2).reshape(4,4,4)
    a=rng.normal(size=(2,2));b=rng.normal(size=(2,2))
    got=np.einsum('ijk,i,j->k',M,a.ravel(),b.ravel())
    index_error=float(la.norm(got-(a@b).ravel()))
    assert index_error<1e-12,index_error
    return dict(jacobian_directional_relative_error=derivative_error,
                rank_one_test_full_relative_residual=full_relative,
                row_major_forward_error=index_error)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input',nargs='?',type=Path)
    parser.add_argument('--out',type=Path,default=Path('data'))
    parser.add_argument('--seed',type=int,default=12345)
    parser.add_argument('--rcond',type=float,default=1e-10)
    parser.add_argument('--ds',type=float,default=1e-4)
    parser.add_argument('--max-ds',type=float,default=.02)
    parser.add_argument('--min-ds',type=float,default=1e-9)
    parser.add_argument('--max-steps',type=int,default=1500)
    parser.add_argument('--tol',type=float,default=1e-11)
    parser.add_argument('--max-newton-iters',type=int,default=20)
    parser.add_argument('--post-turn-steps',type=int,default=30,
                        help='Stop this many accepted steps after the first negative-to-positive t tangent; 0 disables.')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(),indent=2))
        if args.input is None:return
    if args.input is None: parser.error('Provide an input CSV/NPZ, or --self-test.')
    if args.min_ds<=0 or args.ds<args.min_ds or args.max_ds<args.ds or args.tol<=0 or args.rcond<=0:
        parser.error('Require 0 < min-ds <= ds <= max-ds, tol > 0 and rcond > 0.')
    args.out.mkdir(parents=True,exist_ok=False)
    warnings.filterwarnings('error',category=la.LinAlgWarning)
    started=time.perf_counter()
    x0,n,r=read_factors(args.input)
    model=TensorMap(n*n,r);system=ProjectedSystem(model,x0,matmul_tensor(n),args.seed,args.rcond)
    z=np.r_[np.zeros(system.q),1.];tau=system.tangent(z)
    meta=dict(input_path=str(args.input),input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
              command=sys.argv,python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
              model_dimensions=dict(n=n,r=r,parameters=model.size,equations=system.target.size,
                                    projected_equations=system.q,slice_equations=model.size-system.q),
              options={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              reference_normalization=system.target_norm)
    (args.out/'metadata.json').write_text(json.dumps(meta,indent=2))
    np.savez_compressed(args.out/'system.npz',x0=x0,B=system.B,S=system.S,P=system.P,
                        target=system.target,start_tensor=system.f0,singular_values=system.singular_values)
    row=system.diagnostics(z,0,0,0,tau,0.)
    best_t=row.copy();best_rho=row.copy();
    system.save_point(args.out/'best_t.npz',z);system.save_point(args.out/'best_rho.npz',z)
    print(json.dumps({'initial':row,'dimensions':meta['model_dimensions']}),flush=True)
    ds=args.ds;failures=0;turn_step=None;status='max_steps'
    with (args.out/'trace.csv').open('w',newline='') as fh:
        writer=csv.DictWriter(fh,fieldnames=list(row));writer.writeheader();writer.writerow(row)
        for step in range(1,args.max_steps+1):
            while True:
                pred=z+ds*tau
                nxt,it,corr=system.correct(pred,tau,args.tol,args.max_newton_iters)
                if nxt is not None and la.norm(nxt-pred)<=2*ds:
                    try: nxt_tau=system.tangent(nxt,tau)
                    except (la.LinAlgError,la.LinAlgWarning): nxt=None
                else: nxt=None
                if nxt is not None:break
                failures+=1;ds*=.5
                if ds<args.min_ds:
                    status='step_below_min_step';break
            if nxt is None:break
            if turn_step is None and tau[-1]<0 and nxt_tau[-1]>=0:
                turn_step=step
            z,tau=nxt,nxt_tau
            row=system.diagnostics(z,step,ds,it,tau,corr)
            writer.writerow(row);fh.flush()
            if row['t']<best_t['t']:
                best_t=row.copy();system.save_point(args.out/'best_t.npz',z)
            if row['relative_full_residual_to_exact']<best_rho['relative_full_residual_to_exact']:
                best_rho=row.copy();system.save_point(args.out/'best_rho.npz',z)
            if step%50==0:print(json.dumps({k:row[k] for k in ('step','t','ds','relative_full_residual_to_exact','weight')}),flush=True)
            if z[-1]<=0:
                status='crossed_t_zero_no_certification';break
            if args.post_turn_steps and turn_step is not None and step>=turn_step+args.post_turn_steps:
                status='stopped_after_first_turn';break
            if it<=3:ds=min(args.max_ds,ds*1.25)
            elif it>=7:ds=max(args.min_ds,ds*.7)
    system.save_point(args.out/'last.npz',z)
    summary=dict(status=status,first_turn_step=turn_step,rejected_predictors=failures,
                 elapsed_seconds=time.perf_counter()-started,initial_relative_residual=float(la.norm(system.direction)/system.target_norm),
                 minimum_t_point=best_t,minimum_full_relative_residual_point=best_rho,last_point=row)
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
