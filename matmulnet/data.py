"""Training and validation sets: triples (A, B, AB) with entries of A, B uniform in [-1, 1]."""
import numpy as np

N_TRAIN, N_VAL = 10_000, 1_000


def draw_pairs(rng, count, n):
    """Return vec(A), vec(B), vec(AB), each of shape (count, n^2), row-major."""
    A = rng.uniform(-1.0, 1.0, (count, n, n))
    B = rng.uniform(-1.0, 1.0, (count, n, n))
    m = n * n
    return A.reshape(count, m).copy(), B.reshape(count, m).copy(), (A @ B).reshape(count, m).copy()


def draw_sets(rng, n, n_train=N_TRAIN, n_val=N_VAL):
    """Both sets are drawn once and kept fixed for the whole run."""
    return draw_pairs(rng, n_train, n), draw_pairs(rng, n_val, n)
