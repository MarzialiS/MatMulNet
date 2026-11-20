"""Compiled training kernel (numba).

One call runs a whole epoch: for every mini-batch the forward pass
V = ((A H) * (B K)) F, the mean squared error over batch x n^2 entries, its gradient (written
out by hand), global-norm gradient clipping and the Adam update (AdamW with decoupled weight decay when
`decay` = 1 - lr * weight_decay is not 1). The arithmetic follows
torch.nn.utils.clip_grad_norm_ and torch.optim.Adam, so the result agrees with the PyTorch
implementation in `reference_torch.py` up to rounding (see tests/).

The epoch is compiled once per shape (n^2, r, d, batch), with the sizes as constants, and the
mini-batch is the innermost, unit-stride index of the intermediate arrays: this lets LLVM
vectorise every loop. The last, smaller mini-batch is padded with zero samples, which
contribute exactly zero to the loss and to the gradient. error_model="numpy" removes the
division-by-zero checks that would otherwise keep the Adam loop scalar.

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


_EPOCHS = {}


def train_epoch(Hc, Kc, Fc, mH, mK, mF, vH, vK, vF, step, tp, lr, clip, A, B, Y, perm, batch,
                decay=1.0):
    """One pass over the training set in the order `perm`, in mini-batches of `batch` samples
    (the last one may be smaller). Parameters and Adam moments are updated in place.
    Returns (step, training loss averaged over the samples seen)."""
    key = Hc.shape + (batch,)
    if key not in _EPOCHS:
        _EPOCHS[key] = _build_epoch(*key)
    return _EPOCHS[key](Hc, Kc, Fc, mH, mK, mF, vH, vK, vF, step, tp, lr, clip, A, B, Y, perm,
                        decay)


def _build_epoch(m, r, nd, nbt):
    @njit(fastmath=True, boundscheck=False, nogil=True, error_model="numpy")
    def epoch(Hc, Kc, Fc, mH, mK, mF, vH, vK, vF, step, tp, lr, clip, A, B, Y, perm, decay):
        N = perm.shape[0]
        # transposed mini-batch: AT[j, i] = A[perm[start + i], j], and so on
        AT = np.empty((m, nbt))
        BT = np.empty((m, nbt))
        ET = np.empty((m, nbt))
        CT = np.empty((r, nbt))
        DT = np.empty((r, nbt))
        PT = np.empty((r, nbt))
        XDT = np.empty((r, nbt))
        XCT = np.empty((r, nbt))
        H = np.empty((m, r))
        K = np.empty((m, r))
        F = np.empty((r, m))
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
            nb = min(nbt, N - start)
            for i in range(nb):
                ii = perm[start + i]
                for j in range(m):
                    AT[j, i] = A[ii, j]
                    BT[j, i] = B[ii, j]
                    ET[j, i] = -Y[ii, j]
            for i in range(nb, nbt):
                for j in range(m):
                    AT[j, i] = 0.0
                    BT[j, i] = 0.0
                    ET[j, i] = 0.0
            start += nb
            for j in range(m):
                for q in range(r):
                    sh = 0.0
                    sk = 0.0
                    sf = 0.0
                    for d in range(nd):
                        sh += Hf[(j * r + q) * nd + d] * tp[d]
                        sk += Kf[(j * r + q) * nd + d] * tp[d]
                        sf += Ff[(q * m + j) * nd + d] * tp[d]
                    H[j, q] = sh
                    K[j, q] = sk
                    F[q, j] = sf
            # C^T = H^T A^T, D^T = K^T B^T, P^T = C^T * D^T
            for q in range(r):
                for i in range(nbt):
                    CT[q, i] = 0.0
                    DT[q, i] = 0.0
                for j in range(m):
                    h = H[j, q]
                    k_ = K[j, q]
                    for i in range(nbt):
                        CT[q, i] += h * AT[j, i]
                        DT[q, i] += k_ * BT[j, i]
                for i in range(nbt):
                    PT[q, i] = CT[q, i] * DT[q, i]
            # E^T = F^T P^T - Y^T; from here on E holds sc * E = dL/dV
            for k in range(m):
                for q in range(r):
                    f = F[q, k]
                    for i in range(nbt):
                        ET[k, i] += f * PT[q, i]
            sq = 0.0
            sc = 2.0 / (nb * m)
            for k in range(m):
                for i in range(nbt):
                    e = ET[k, i]
                    sq += e * e
                    ET[k, i] = sc * e
            loss_sum += sq / m
            # dL/dF = P^T E; dL/dP = E F^T, times D and C for the gradients of H and K
            for q in range(r):
                for k in range(m):
                    s = 0.0
                    for i in range(nbt):
                        s += PT[q, i] * ET[k, i]
                    dF[q, k] = s
                for i in range(nbt):
                    XDT[q, i] = 0.0
                for k in range(m):
                    f = F[q, k]
                    for i in range(nbt):
                        XDT[q, i] += f * ET[k, i]
                for i in range(nbt):
                    x = XDT[q, i]
                    XDT[q, i] = x * DT[q, i]
                    XCT[q, i] = x * CT[q, i]
            # dL/dH = A^T (X * D), dL/dK = B^T (X * C)
            for j in range(m):
                for q in range(r):
                    s = 0.0
                    t = 0.0
                    for i in range(nbt):
                        s += AT[j, i] * XDT[q, i]
                        t += BT[j, i] * XCT[q, i]
                    dH[j, q] = s
                    dK[j, q] = t
            # clipping on the global norm of the gradient w.r.t. the coefficients
            # (d coeff = d factor * tp)
            g2 = 0.0
            for j in range(m):
                for q in range(r):
                    g2 += dH[j, q] * dH[j, q] + dK[j, q] * dK[j, q] + dF[q, j] * dF[q, j]
            coef = clip / (np.sqrt(g2 * tp2) + 1e-6)
            if coef > 1.0:
                coef = 1.0
            step += 1.0
            step_size = lr / (1.0 - BETA1 ** step)
            inv_bc2_sqrt = 1.0 / np.sqrt(1.0 - BETA2 ** step)
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
            _adam(Hf, mHf, vHf, gH, step_size, inv_bc2_sqrt, decay)
            _adam(Kf, mKf, vKf, gK, step_size, inv_bc2_sqrt, decay)
            _adam(Ff, mFf, vFf, gF, step_size, inv_bc2_sqrt, decay)
        return step, loss_sum / N

    return epoch


@njit(cache=True, fastmath=True, boundscheck=False, nogil=True, error_model="numpy")
def _adam(p, m, v, g, step_size, inv_bc2_sqrt, decay):
    for i in range(p.size):
        gi = g[i]
        mi = BETA1 * m[i] + (1.0 - BETA1) * gi
        vi = BETA2 * v[i] + (1.0 - BETA2) * gi * gi
        m[i] = mi
        v[i] = vi
        p[i] = p[i] * decay - step_size * mi / (np.sqrt(vi) * inv_bc2_sqrt + ADAM_EPS)
