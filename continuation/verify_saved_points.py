#!/usr/bin/env python3
"""Independent full-tensor checks of the saved continuation points, in extended precision.

    python verify_saved_points.py [data]
"""
from pathlib import Path
import sys
import json
import numpy as np
from scipy import linalg as la
from threadpoolctl import threadpool_limits
from continuation import read_factors,TensorMap,matmul_tensor,self_test


def tensor_loop(H,K,F,dtype=np.longdouble):
    H=np.asarray(H,dtype=dtype);K=np.asarray(K,dtype=dtype);F=np.asarray(F,dtype=dtype)
    out=np.zeros((H.shape[0],K.shape[0],F.shape[1]),dtype=dtype)
    for s in range(H.shape[1]):
        out += H[:,s,None,None]*K[None,:,s,None]*F[s,None,None,:]
    return out.ravel()


def norm(a):return np.sqrt(np.sum(a*a,dtype=np.longdouble))


def main():
    root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent/'data'
    with np.load(root/'system.npz') as d:
        target=d['target'];start=d['start_tensor'];P=d['P'];B=d['B'];S=d['S'];x0=d['x0']
    normM=norm(target.astype(np.longdouble))
    output={'unit_tests':self_test(),
            'projection_row_orthogonality_error':float(la.norm(P@P.T-np.eye(P.shape[0]))),
            'slice_basis_orthogonality_error':float(la.norm(B.T@B-np.eye(B.shape[1]))),
            'longdouble_precision_bits':np.finfo(np.longdouble).nmant,
            'saved_point_checks':{}}
    for name in ('best_t','best_rho','last'):
        with np.load(root/f'{name}.npz') as d:
            H,K,F,t,y=[d[k] for k in ('H','K','F','t','y')]
        value=tensor_loop(H,K,F)
        x=np.r_[H.T.ravel(),K.T.ravel(),F.ravel()]
        current=target.astype(np.longdouble)+np.longdouble(t)*(start.astype(np.longdouble)-target)
        exact=norm(value-target)
        output['saved_point_checks'][name]={
            't':float(t),'full_relative_residual_extended':float(exact/normM),
            'full_residual_to_current_target_extended':float(norm(value-current)),
            'projected_residual_absolute_extended':float(norm(P.astype(np.longdouble)@(value-current))),
            'slice_residual':float(la.norm(S@(x-x0))),
            'saved_coordinate_consistency':float(la.norm(x-(x0+B@y))),
            'weight':float(np.sum(norms:=np.prod([la.norm(A,axis=0) for A in (H,K,F.T)],axis=0))),
            'max_term_scale':float(norms.max())}
    (root/'verification.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
