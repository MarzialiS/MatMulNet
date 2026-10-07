"""Constrained refinement of a trained decomposition (projected Levenberg-Marquardt).

    python -m refinement.refine --init runs/rank_n3_r23_seed1/best.npz --out refinement_r23

For an increasing sequence of bounds c, minimise the full-tensor residual

    rho(theta) = ||T_theta - M_n||_F / ||M_n||_F     subject to   |theta_i| <= c,

where theta collects all entries of H, K, F. Each level is warm-started from the solution of
the previous one, so the levels form a chain along which rho cannot increase. The chain stops
as soon as rho falls below the double-precision cancellation level of the current solution.

Output in --out: frontier.csv (one row per level: c, rho, weight W, largest |theta_i|, number
of coefficients at the bound, cancellation level, iterations, status) and level_<c>.npz with
the factors reached at each level.
"""
import argparse
import csv
import os
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


def projected_lm(prob, x, c, max_iter=50_000, kkt_tol=1e-9, stall_window=200, stall_tol=1e-7):
    """Levenberg-Marquardt on the box |x_i| <= c. The step is computed on the variables that
    are free at the current point (not at a bound with the gradient pointing outwards), the
    trial point is projected back onto the box, and the damping is driven by the ratio of the
    achieved to the predicted reduction. Returns (x, status, iterations)."""
    x = np.clip(x, -c, c)
    res = prob.residual(x)
    f = 0.5 * res @ res
    mu, history = 1e-3, [np.sqrt(2 * f)]
    for it in range(1, max_iter + 1):
        H, K, F = prob.unpack(x)
        if np.sqrt(2 * f) < metrics.cancellation_level(H, K, F, prob.n):
            return x, "machine_precision", it - 1
        J = prob.jacobian(x)
        g = J.T @ res
        if np.max(np.abs(x - np.clip(x - g, -c, c))) < kkt_tol * np.sqrt(2 * f):
            return x, "kkt", it - 1
        free = ~(((x >= c) & (g < 0)) | ((x <= -c) & (g > 0)))
        Jf = J[:, free]
        A = Jf.T @ Jf
        diag = np.maximum(np.diag(A), 1e-12)
        while True:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", la.LinAlgWarning)
                    step = la.solve(A + mu * np.diag(diag), -g[free], assume_a="pos")
            except (la.LinAlgError, la.LinAlgWarning):      # singular to working precision
                mu *= 10.0
                if mu > 1e15:
                    return _polish(prob, x, c, "stalled", it)
                continue
            trial = x.copy()
            trial[free] += step
            trial = np.clip(trial, -c, c)
            d = (trial - x)[free]
            predicted = -(g[free] @ d + 0.5 * d @ (A @ d))
            res_t = prob.residual(trial)
            f_t = 0.5 * res_t @ res_t
            if predicted > 0 and f_t < f and (f - f_t) / predicted > 1e-4:
                gain = (f - f_t) / predicted
                x, res, f = trial, res_t, f_t
                mu = max(mu * (0.2 if gain > 0.75 else 1.0 if gain > 0.25 else 4.0), 1e-15)
                break
            mu *= 4.0
            if mu > 1e15:
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


def chain(H, K, F, n, out, levels=LEVELS, max_iter=50_000, verbose=True):
    prob = Problem(n, H.shape[1])
    x = prob.pack(H, K, F)
    os.makedirs(out, exist_ok=True)
    rows = []
    with open(os.path.join(out, "frontier.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["c", "rho", "W", "max_abs_theta", "at_bound", "cancellation_level",
                     "iterations", "status"])
        for c in levels:
            x, status, iters = projected_lm(prob, x, float(c), max_iter)
            Hc, Kc, Fc = prob.unpack(x)
            row = [c, metrics.rho(Hc, Kc, Fc, n), metrics.weight(Hc, Kc, Fc),
                   float(np.max(np.abs(x))), int(np.sum(np.abs(x) >= c)),
                   metrics.cancellation_level(Hc, Kc, Fc, n), iters, status]
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
    a = ap.parse_args()
    z = np.load(a.init)
    H, K, F = (np.asarray(z[k], dtype=np.float64) for k in ("H", "K", "F"))
    with threadpool_limits(limits=1):       # the linear systems are small: one thread is fastest
        chain(H, K, F, int(round(np.sqrt(H.shape[0]))), a.out, a.levels, a.max_iter)


if __name__ == "__main__":
    main()
