"""Compiled training kernel (numba).

One call runs a whole epoch: for every mini-batch the forward pass
V = ((A H) * (B K)) F, the mean squared error over batch x n^2 entries, its gradient (written
out by hand), global-norm gradient clipping and the Adam update. The arithmetic follows
torch.nn.utils.clip_grad_norm_ and torch.optim.Adam, so the result agrees with the PyTorch
implementation in `reference_torch.py` up to rounding (see tests/).

Parameters are coefficient tensors Hc, Kc of shape (n^2, r, d) and Fc of shape (r, n^2, d),
with H = sum_e tp[e] Hc[:, :, e] and tp[e] = eps^(d_min + e). The rank model is the case d = 1,
tp = [1]. All arrays are float64 and C-contiguous.
"""
import numpy as np
from numba import njit

BETA1, BETA2, ADAM_EPS = 0.9, 0.999, 1e-8


@njit(cache=True, boundscheck=False, nogil=True)
def _evaluate(P, tp, out):
    n0, n1, nd = P.shape
    for a in range(n0):
        for b in range(n1):
            s = 0.0
            for d in range(nd):
                s += P[a, b, d] * tp[d]
            out[a, b] = s


@njit(cache=True, fastmath=True, boundscheck=False, nogil=True)
def _forward(H, K, F, A, B, Y, nb, C, D, P, E):
    """Fill C = A H, D = B K, P = C * D, E = P F - Y for the first nb rows; return sum E^2."""
    m, r = H.shape
    tot = 0.0
    for i in range(nb):
        for q in range(r):
            C[i, q] = 0.0
            D[i, q] = 0.0
        for j in range(m):
            a = A[i, j]
            b = B[i, j]
            for q in range(r):
                C[i, q] += a * H[j, q]
                D[i, q] += b * K[j, q]
        for q in range(r):
            P[i, q] = C[i, q] * D[i, q]
        for k in range(m):
            E[i, k] = -Y[i, k]
        for q in range(r):
            p = P[i, q]
            for k in range(m):
                E[i, k] += p * F[q, k]
        for k in range(m):
            tot += E[i, k] * E[i, k]
    return tot


@njit(cache=True, fastmath=True, boundscheck=False, nogil=True)
def mse(Hc, Kc, Fc, tp, A, B, Y):
    """Mean squared error per output entry over the whole set (A, B, Y)."""
    m, r, _ = Hc.shape
    H = np.empty((m, r))
    K = np.empty((m, r))
    F = np.empty((r, m))
    _evaluate(Hc, tp, H)
    _evaluate(Kc, tp, K)
    _evaluate(Fc, tp, F)
    n = A.shape[0]
    C = np.empty((n, r))
    D = np.empty((n, r))
    P = np.empty((n, r))
    E = np.empty((n, m))
    return _forward(H, K, F, A, B, Y, n, C, D, P, E) / (n * m)


@njit(cache=True, fastmath=True, boundscheck=False, nogil=True)
def _adam(p, m, v, g, step_size, bc2_sqrt):
    for i in range(p.size):
        gi = g[i]
        mi = BETA1 * m[i] + (1.0 - BETA1) * gi
        vi = BETA2 * v[i] + (1.0 - BETA2) * gi * gi
        m[i] = mi
        v[i] = vi
        p[i] -= step_size * mi / (np.sqrt(vi) / bc2_sqrt + ADAM_EPS)


@njit(cache=True, fastmath=True, boundscheck=False, nogil=True)
def train_epoch(Hc, Kc, Fc, mH, mK, mF, vH, vK, vF, step, tp, lr, clip, A, B, Y, perm, batch):
    """One pass over the training set in the order `perm`, in mini-batches of `batch` samples
    (the last one may be smaller). Parameters and Adam moments are updated in place.
    Returns (step, training loss averaged over the samples seen)."""
    m, r, nd = Hc.shape
    N = perm.shape[0]
    H = np.empty((m, r))
    K = np.empty((m, r))
    F = np.empty((r, m))
    C = np.empty((batch, r))
    D = np.empty((batch, r))
    P = np.empty((batch, r))
    E = np.empty((batch, m))
    X = np.empty((batch, r))
    XD = np.empty((batch, r))
    XC = np.empty((batch, r))
    Ab = np.empty((batch, m))
    Bb = np.empty((batch, m))
    Yb = np.empty((batch, m))
    dH = np.empty((m, r))
    dK = np.empty((m, r))
    dF = np.empty((r, m))
    gH = np.empty(Hc.size)
    gK = np.empty(Kc.size)
    gF = np.empty(Fc.size)
    Hf, Kf, Ff = Hc.reshape(-1), Kc.reshape(-1), Fc.reshape(-1)
    mHf, mKf, mFf = mH.reshape(-1), mK.reshape(-1), mF.reshape(-1)
    vHf, vKf, vFf = vH.reshape(-1), vK.reshape(-1), vF.reshape(-1)
    tp2 = 0.0
    for d in range(nd):
        tp2 += tp[d] * tp[d]
    loss_sum = 0.0
    start = 0
    while start < N:
        nb = min(batch, N - start)
        for i in range(nb):
            ii = perm[start + i]
            for j in range(m):
                Ab[i, j] = A[ii, j]
                Bb[i, j] = B[ii, j]
                Yb[i, j] = Y[ii, j]
        start += nb
        _evaluate(Hc, tp, H)
        _evaluate(Kc, tp, K)
        _evaluate(Fc, tp, F)
        sq = _forward(H, K, F, Ab, Bb, Yb, nb, C, D, P, E)
        loss_sum += sq / m
        sc = 2.0 / (nb * m)
        # dL/dF = sc P^T E
        for q in range(r):
            for k in range(m):
                dF[q, k] = 0.0
        for i in range(nb):
            for q in range(r):
                p = sc * P[i, q]
                for k in range(m):
                    dF[q, k] += p * E[i, k]
        # dL/dP = sc E F^T; dL/dH = A^T (dL/dP * D), dL/dK = B^T (dL/dP * C)
        for i in range(nb):
            for q in range(r):
                X[i, q] = 0.0
            for k in range(m):
                e = sc * E[i, k]
                for q in range(r):
                    X[i, q] += e * F[q, k]
        for j in range(m):
            for q in range(r):
                dH[j, q] = 0.0
                dK[j, q] = 0.0
        for i in range(nb):
            for q in range(r):
                XD[i, q] = X[i, q] * D[i, q]
                XC[i, q] = X[i, q] * C[i, q]
            for j in range(m):
                a = Ab[i, j]
                b = Bb[i, j]
                for q in range(r):
                    dH[j, q] += a * XD[i, q]
                    dK[j, q] += b * XC[i, q]
        # clipping on the global norm of the gradient w.r.t. the coefficients (d coeff = d factor * tp)
        g2 = 0.0
        for j in range(m):
            for q in range(r):
                g2 += dH[j, q] * dH[j, q] + dK[j, q] * dK[j, q] + dF[q, j] * dF[q, j]
        coef = clip / (np.sqrt(g2 * tp2) + 1e-6)
        if coef > 1.0:
            coef = 1.0
        for j in range(m):
            for q in range(r):
                o = (j * r + q) * nd
                o2 = (q * m + j) * nd
                h = dH[j, q] * coef
                k_ = dK[j, q] * coef
                f = dF[q, j] * coef
                for d in range(nd):
                    gH[o + d] = h * tp[d]
                    gK[o + d] = k_ * tp[d]
                    gF[o2 + d] = f * tp[d]
        step += 1.0
        step_size = lr / (1.0 - BETA1 ** step)
        bc2_sqrt = np.sqrt(1.0 - BETA2 ** step)
        _adam(Hf, mHf, vHf, gH, step_size, bc2_sqrt)
        _adam(Kf, mKf, vKf, gK, step_size, bc2_sqrt)
        _adam(Ff, mFf, vFf, gF, step_size, bc2_sqrt)
    return step, loss_sum / N
