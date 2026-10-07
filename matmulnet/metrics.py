"""Tensor-level quantities: the matrix-multiplication tensor, the residual rho and the weight W.

Conventions. A matrix is flattened row-major, so entry (i, j) of an n x n matrix has index
n*i + j. The factors are H, K of shape (n^2, r) and F of shape (r, n^2); summand s is
(h_s, k_s, f_s) = (H[:, s], K[:, s], F[s, :]) and the represented tensor is

    T[a, b, c] = sum_s H[a, s] K[b, s] F[s, c],

so that vec(AB) = F^T ((H^T vec A) * (K^T vec B)) whenever T equals the multiplication tensor.
"""
import numpy as np


def matmul_tensor(n):
    """M_n with M[n*i+j, n*j+k, n*i+k] = 1: sum_ab M[a, b, c] vec(A)_a vec(B)_b = vec(AB)_c."""
    M = np.zeros((n * n, n * n, n * n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                M[n * i + j, n * j + k, n * i + k] = 1.0
    return M


def tensor(H, K, F):
    return np.einsum("as,bs,sc->abc", H, K, F, optimize=True)


def rho(H, K, F, n):
    """Relative residual ||T - M_n||_F / ||M_n||_F."""
    M = matmul_tensor(n)
    return float(np.linalg.norm(tensor(H, K, F) - M) / np.linalg.norm(M))


def weight(H, K, F):
    """W = sum_s ||h_s|| ||k_s|| ||f_s||, invariant under termwise rescaling of the summands."""
    return float(np.sum(np.linalg.norm(H, axis=0) * np.linalg.norm(K, axis=0)
                        * np.linalg.norm(F, axis=1)))


def cancellation_level(H, K, F, n):
    """Size of the double-precision rounding error on rho for this decomposition:
    machine epsilon times ||sum_s |h_s| x |k_s| x |f_s|||_F / ||M_n||_F."""
    A = tensor(np.abs(H), np.abs(K), np.abs(F))
    return float(np.finfo(np.float64).eps * np.linalg.norm(A) / np.linalg.norm(matmul_tensor(n)))


def evaluate(coeff, degrees, eps):
    """Evaluate a Laurent factor: sum_e eps^e coeff[..., e - d_min]."""
    return coeff @ (float(eps) ** np.asarray(degrees, dtype=np.float64))
