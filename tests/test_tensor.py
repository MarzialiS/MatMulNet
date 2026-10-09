import numpy as np

from matmulnet import metrics
from refinement.refine import Problem, chain


def naive(n):
    """The standard decomposition with n^3 multiplications a_ij * b_jk."""
    m, r = n * n, n ** 3
    H, K, F = np.zeros((m, r)), np.zeros((m, r)), np.zeros((r, m))
    for s, (i, j, k) in enumerate(np.ndindex(n, n, n)):
        H[n * i + j, s] = K[n * j + k, s] = F[s, n * i + k] = 1.0
    return H, K, F


def test_matmul_tensor_multiplies():
    rng = np.random.default_rng(0)
    for n in (2, 3):
        A, B = rng.normal(size=(n, n)), rng.normal(size=(n, n))
        got = np.einsum("abc,a,b->c", metrics.matmul_tensor(n), A.ravel(), B.ravel())
        assert np.allclose(got, (A @ B).ravel())
        H, K, F = naive(n)
        assert metrics.rho(H, K, F, n) == 0.0
        assert np.allclose(((A.ravel() @ H) * (B.ravel() @ K)) @ F, (A @ B).ravel())
        assert metrics.weight(H, K, F) == n ** 3


def test_jacobian():
    rng = np.random.default_rng(1)
    prob = Problem(2, 7)
    x, d = rng.normal(size=3 * 4 * 7), rng.normal(size=3 * 4 * 7)
    fd = (prob.residual(x + 1e-6 * d) - prob.residual(x - 1e-6 * d)) / 2e-6
    assert np.allclose(prob.jacobian(x) @ d, fd, rtol=1e-6, atol=1e-8)
    J, res = prob.jacobian(x), prob.residual(x)
    A, g = prob.normal(x, res)
    assert np.allclose(A, J.T @ J, rtol=1e-12, atol=1e-12) and np.allclose(g, J.T @ res)


def test_refinement_reaches_machine_precision(tmp_path):
    rng = np.random.default_rng(2)
    H, K, F = (x + 0.05 * rng.normal(size=x.shape) for x in naive(2))
    rows = chain(H, K, F, 2, str(tmp_path), levels=[0.2, 1.5], verbose=False)
    assert rows[0][1] > 1e-3            # the box |theta| <= 0.2 is too small
    assert rows[-1][1] < 1e-13


def test_strassen_and_its_square():
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location(
        "known_schemes", pathlib.Path(__file__).parents[1] / "experiments" / "known_schemes.py")
    ks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ks)
    assert metrics.rho(*ks.strassen(), 2) == 0.0
    H, K, F = ks.kronecker(ks.strassen(), ks.strassen(), 2, 2)
    assert H.shape == (16, 49) and metrics.rho(H, K, F, 4) == 0.0
    lau = metrics.laurent(H[..., None], K[..., None], F[..., None], [0])
    assert list(lau) == [0] and np.array_equal(lau[0], metrics.matmul_tensor(4))
