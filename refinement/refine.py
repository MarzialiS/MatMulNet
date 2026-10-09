"""Constrained refinement of a trained decomposition (projected Levenberg-Marquardt).

    python -m refinement.refine --init runs/rank_n3_r23_seed1/best.npz --out refinement_r23

For an increasing sequence of bounds c, minimise the full-tensor residual

    rho(theta) = ||T_theta - M_n||_F / ||M_n||_F     subject to   |theta_i| <= c,

where theta collects all entries of H, K, F. Each level is warm-started from the solution of
the previous one, so the levels form a chain along which rho cannot increase. The chain stops
as soon as rho falls below the double-precision cancellation level of the current solution.

Output in --out: frontier.csv (one row per level: c, rho, weight W, largest |theta_i|, number
of coefficients at the bound, cancellation level, iterations, status, time) and level_<c>.npz
with the factors reached at each level. --max-seconds bounds the time spent on each level.
"""
import argparse
import csv
import os
import time
import warnings

import numpy as np
from scipy import linalg as la
from threadpoolctl import threadpool_limits

from matmulnet import metrics

LEVELS = [round(0.1 * i, 1) for i in range(1, 10)] + list(range(1, 26))


class Problem:
    def __init__(self, n, r):
        self.n, self.m, self.r = n, n * n, r
        self.M = metrics.matmul_tensor(n)
        self.normM = float(np.linalg.norm(self.M))
        self.eye = np.eye(self.m)

    def unpack(self, x):
        m, r = self.m, self.r
        return x[:m * r].reshape(m, r), x[m * r:2 * m * r].reshape(m, r), x[2 * m * r:].reshape(r, m)

    @staticmethod
    def pack(H, K, F):
        return np.concatenate([H.ravel(), K.ravel(), F.ravel()])

    def residual(self, x):
        """(T_theta - M_n) / ||M_n||, flattened: its Euclidean norm is rho."""
        return (metrics.tensor(*self.unpack(x)) - self.M).ravel() / self.normM

    def jacobian(self, x):
        H, K, F = self.unpack(x)
        m, r, I = self.m, self.r, self.eye
        return np.concatenate([
            np.einsum("ai,bs,sc->abcis", I, K, F).reshape(m ** 3, m * r),
            np.einsum("as,bi,sc->abcis", H, I, F).reshape(m ** 3, m * r),
            np.einsum("as,bs,ck->abcsk", H, K, I).reshape(m ** 3, m * r)], axis=1) / self.normM

    def normal(self, x, res):
        """J^T J and J^T res without forming J: every block of J^T J is a product of the
        Gram matrices of two factors and of entries of the third."""
        H, K, F = self.unpack(x)
        m, r, I = self.m, self.r, self.eye
        hh, kk, ff = H.T @ H, K.T @ K, F @ F.T
        A = np.empty((3 * m * r, 3 * m * r))
        o = m * r
        A[:o, :o] = np.einsum("ab,st->asbt", I, kk * ff).reshape(o, o)
        A[o:2 * o, o:2 * o] = np.einsum("ab,st->asbt", I, hh * ff).reshape(o, o)
        A[2 * o:, 2 * o:] = np.einsum("st,cd->sctd", hh * kk, I).reshape(o, o)
        A[:o, o:2 * o] = np.einsum("bs,at,st->asbt", K, H, ff).reshape(o, o)
        A[:o, 2 * o:] = np.einsum("st,sc,at->astc", kk, F, H).reshape(o, o)
        A[o:2 * o, 2 * o:] = np.einsum("st,sc,bt->bstc", hh, F, K).reshape(o, o)
        A[o:2 * o, :o] = A[:o, o:2 * o].T
        A[2 * o:, :o] = A[:o, 2 * o:].T
        A[2 * o:, o:2 * o] = A[o:2 * o, 2 * o:].T
        R = res.reshape(m, m, m)
        g = self.pack(np.einsum("abc,bs,sc->as", R, K, F, optimize=True),
                      np.einsum("abc,as,sc->bs", R, H, F, optimize=True),
                      np.einsum("abc,as,bs->sc", R, H, K, optimize=True))
        return A / self.normM ** 2, g / self.normM


def projected_lm(prob, x, c, max_iter=50_000, kkt_tol=1e-9, stall_window=200, stall_tol=1e-7,
                 max_seconds=None):
    """Levenberg-Marquardt on the box |x_i| <= c. The step is computed on the variables that
    are free at the current point (not at a bound with the gradient pointing outwards), the
    trial point is projected back onto the box, and the damping is driven by the ratio of the
    achieved to the predicted reduction. Returns (x, status, iterations); the status is
    "time_limit" when max_seconds elapse first."""
    x = np.clip(x, -c, c)
    res = prob.residual(x)
    f = 0.5 * res @ res
    mu, history, t0 = 1e-3, [np.sqrt(2 * f)], time.time()
    for it in range(1, max_iter + 1):
        if max_seconds is not None and time.time() - t0 > max_seconds:
            return _polish(prob, x, c, "time_limit", it - 1)
        H, K, F = prob.unpack(x)
        if np.sqrt(2 * f) < metrics.cancellation_level(H, K, F, prob.n):
            return x, "machine_precision", it - 1
        if prob.n <= 3:                 # small enough to form the Jacobian
            J = prob.jacobian(x)
            g = J.T @ res
        else:
            N, g = prob.normal(x, res)
        if np.max(np.abs(x - np.clip(x - g, -c, c))) < kkt_tol * np.sqrt(2 * f):
            return x, "kkt", it - 1
        free = ~(((x >= c) & (g < 0)) | ((x <= -c) & (g > 0)))
        if prob.n <= 3:
            Jf = J[:, free]
            A = Jf.T @ Jf
        else:
            A = N[np.ix_(free, free)]
        diag = np.maximum(np.diag(A), 1e-12)
        # damping search: first upwards from the current value; if the predicted reduction
        # drops below what double precision can resolve, downwards from it
        mu_try, direction, accepted = mu, 4.0, False
        while True:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", la.LinAlgWarning)
                    step = la.solve(A + mu_try * np.diag(diag), -g[free], assume_a="pos")
            except (la.LinAlgError, la.LinAlgWarning):      # singular to working precision
                step = None
            if step is not None:
                trial = x.copy()
                trial[free] += step
                trial = np.clip(trial, -c, c)
                d = (trial - x)[free]
                predicted = -(g[free] @ d + 0.5 * d @ (A @ d))
                res_t = prob.residual(trial)
                f_t = 0.5 * res_t @ res_t
                if predicted > 0 and f_t < f and (f - f_t) / predicted > 1e-4:
                    gain = (f - f_t) / predicted
                    x, res, f, accepted = trial, res_t, f_t, True
                    mu = max(mu_try * (0.2 if gain > 0.75 else 1.0 if gain > 0.25 else 4.0), 1e-14)
                    break
            if direction > 1 and (mu_try > 1e15 or (step is not None and 0 <= predicted < 1e-15 * f)):
                mu_try, direction = mu, 0.25
            mu_try *= direction
            if mu_try < 1e-14:
                break
        if not accepted:
            return _polish(prob, x, c, "stalled", it)
        history.append(np.sqrt(2 * f))
        if len(history) > stall_window and history[-1] > history[-1 - stall_window] * (1 - stall_tol):
            return _polish(prob, x, c, "stalled", it)
    return x, "max_iter", max_iter


def _polish(prob, x, c, status, it, rounds=20):
    """Close to a zero of the residual the damped iteration stops at the rounding level of its
    own acceptance test. A few undamped Gauss-Newton steps, kept only when they lower rho,
    bring the point down to the cancellation level."""
    rho = np.linalg.norm(prob.residual(x))
    if rho > 1e-8:
        return x, status, it
    for _ in range(rounds):
        if rho < metrics.cancellation_level(*prob.unpack(x), prob.n):
            return x, "machine_precision", it
        free = np.abs(x) < c                        # coefficients at the bound stay there
        trial = x.copy()
        # the Jacobian is rank deficient (termwise rescalings): truncated least squares
        trial[free] += la.lstsq(prob.jacobian(x)[:, free], -prob.residual(x), cond=1e-10)[0]
        trial = np.clip(trial, -c, c)
        rho_t = np.linalg.norm(prob.residual(trial))
        if rho_t >= rho:
            break
        x, rho = trial, rho_t
    return x, status, it


def chain(H, K, F, n, out, levels=LEVELS, max_iter=50_000, verbose=True, max_seconds=None):
    prob = Problem(n, H.shape[1])
    x = prob.pack(H, K, F)
    os.makedirs(out, exist_ok=True)
    rows = []
    with open(os.path.join(out, "frontier.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["c", "rho", "W", "max_abs_theta", "at_bound", "cancellation_level",
                     "iterations", "status", "seconds"])
        for c in levels:
            t0 = time.time()
            x, status, iters = projected_lm(prob, x, float(c), max_iter, max_seconds=max_seconds)
            Hc, Kc, Fc = prob.unpack(x)
            row = [c, metrics.rho(Hc, Kc, Fc, n), metrics.weight(Hc, Kc, Fc),
                   float(np.max(np.abs(x))), int(np.sum(np.abs(x) >= c)),
                   metrics.cancellation_level(Hc, Kc, Fc, n), iters, status,
                   round(time.time() - t0, 1)]
            wr.writerow([f"{v:.9e}" if isinstance(v, float) else v for v in row])
            fh.flush()
            np.savez(os.path.join(out, f"level_{c}.npz"), H=Hc, K=Kc, F=Fc, c=c)
            rows.append(row)
            if verbose:
                print(f"c = {c:<4}  rho = {row[1]:.3e}  W = {row[2]:.4e}  {status} ({iters} it)",
                      flush=True)
            if status == "machine_precision":
                break
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", required=True, help=".npz with the factors H, K, F (e.g. best.npz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--levels", type=float, nargs="+", default=LEVELS)
    ap.add_argument("--max-iter", type=int, default=50_000)
    ap.add_argument("--max-seconds", type=float, help="time limit per level")
    ap.add_argument("--threads", type=int, default=1, help="threads of the linear algebra")
    a = ap.parse_args()
    z = np.load(a.init)
    H, K, F = (np.asarray(z[k], dtype=np.float64) for k in ("H", "K", "F"))
    with threadpool_limits(limits=a.threads):   # one thread is fastest for 3 x 3
        chain(H, K, F, int(round(np.sqrt(H.shape[0]))), a.out, a.levels, a.max_iter,
              max_seconds=a.max_seconds)


if __name__ == "__main__":
    main()
