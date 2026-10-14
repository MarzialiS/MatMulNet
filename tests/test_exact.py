"""The exact loss: its value, its gradient, and a training run that resumes exactly."""
import numpy as np
import pytest

from matmulnet import metrics
from matmulnet.data import draw_sets
from matmulnet.exact import exact_loss, loss_grad, rho_of_loss, train_exact


@pytest.mark.parametrize("n,r", [(2, 7), (3, 23), (4, 48)])
def test_loss_is_rho(n, r):
    rng = np.random.default_rng(n)
    H, K, F = rng.standard_normal((n * n, r)), rng.standard_normal((n * n, r)), \
        rng.standard_normal((r, n * n))
    loss = exact_loss(H, K, F, metrics.matmul_tensor(n))
    assert abs(rho_of_loss(loss, n) / metrics.rho(H, K, F, n) - 1) < 1e-12


def test_loss_is_expected_sampled_loss():
    """The exact loss is the expectation of the per-entry squared error on random (A, B)."""
    n, r = 2, 6
    rng = np.random.default_rng(1)
    H, K, F = 0.5 * rng.standard_normal((4, r)), 0.5 * rng.standard_normal((4, r)), \
        0.5 * rng.standard_normal((r, 4))
    (A, B, Y), _ = draw_sets(rng, n, 400_000, 1)
    sampled = np.mean((((A @ H) * (B @ K)) @ F - Y) ** 2)
    assert abs(sampled / exact_loss(H, K, F, metrics.matmul_tensor(n)) - 1) < 1e-2


def test_gradient():
    n, r = 3, 5
    rng = np.random.default_rng(2)
    P = [rng.standard_normal((9, r)), rng.standard_normal((9, r)), rng.standard_normal((r, 9))]
    M = metrics.matmul_tensor(n)
    G = [np.empty_like(p) for p in P]
    loss_grad(*P, M, *G)
    h = 1e-6
    for p, g in zip(P, G):
        for idx in [(0, 0), (3, 2), (8, 4)] if p.shape[0] == 9 else [(0, 0), (2, 3), (4, 8)]:
            old = p[idx]
            p[idx] = old + h
            up = exact_loss(*P, M)
            p[idx] = old - h
            down = exact_loss(*P, M)
            p[idx] = old
            assert abs((up - down) / (2 * h) - g[idx]) < 1e-6 * max(1.0, abs(g[idx]))


def test_resume_is_exact(tmp_path):
    kw = dict(seed=3, steps=3000, lr=1e-2, chunk=100, log_every=500, ckpt_every=1000, quiet=True,
              stall=0)
    a = train_exact(2, 7, str(tmp_path / "a"), **kw)
    train_exact(2, 7, str(tmp_path / "b"), _interrupt_at=20, **kw)
    b = train_exact(2, 7, str(tmp_path / "b"), **kw)
    assert a["loss"] == b["loss"] and a["step"] == b["step"]
    za, zb = np.load(tmp_path / "a" / "ckpt.npz"), np.load(tmp_path / "b" / "ckpt.npz")
    assert all(np.array_equal(za[k], zb[k]) for k in ("p0", "p1", "p2", "m0", "v0", "b0"))
    assert (tmp_path / "a" / "per_step.csv").read_text() == \
        (tmp_path / "b" / "per_step.csv").read_text()


def test_strassen_from_noise(tmp_path):
    """2 x 2, rank 7: a short run from a random point reaches an exact decomposition."""
    best = None
    for seed in range(1, 6):
        best = train_exact(2, 7, str(tmp_path / f"s{seed}"), seed=seed, steps=100_000, lr=1e-2,
                           lr_min=1e-8, patience=2000, quiet=True)
        if rho_of_loss(best["loss"], 2) < 1e-8:
            break
    assert rho_of_loss(best["loss"], 2) < 1e-8
