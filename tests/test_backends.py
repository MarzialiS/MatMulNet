"""The compiled kernel and the PyTorch reference must produce the same training trajectory."""
import numpy as np
import pytest

from matmulnet.data import draw_sets
from matmulnet.train import BATCH, init_params, make_backend


@pytest.mark.parametrize("n,r,deg,eps,scale,wd", [(2, 7, [0], 1.0, 1.0, 0.0),
                                                  (3, 23, [0], 1.0, 1.0, 0.0),
                                                  (3, 19, [-2, -1, 0, 1, 2, 3, 4], 0.1, 0.01, 0.0),
                                                  (3, 23, [0], 1.0, 1.0, 50.0)])
def test_same_trajectory(n, r, deg, eps, scale, wd):
    rng = np.random.default_rng(0)
    train, val = draw_sets(rng, n, 500, 100)
    params = [scale * p for p in init_params(rng, n, r, deg)]
    nb = make_backend("numba", params, deg, weight_decay=wd)
    tc = make_backend("torch", params, deg, weight_decay=wd)
    assert abs(nb.validate(eps, val) / tc.validate(eps, val) - 1) < 1e-12
    for _ in range(3):
        perm = rng.permutation(500)
        la, lb = nb.train_epoch(eps, 1e-6, train, perm, BATCH), tc.train_epoch(eps, 1e-6, train, perm, BATCH)
        assert abs(la / lb - 1) < 1e-9
    for p, q in zip(nb.get_state()[0], tc.get_state()[0]):
        assert np.max(np.abs(p - q)) < 1e-10 * max(1.0, np.max(np.abs(q)))
    for a, b in zip(nb.get_state()[1] + nb.get_state()[2], tc.get_state()[1] + tc.get_state()[2]):
        assert np.allclose(a, b, rtol=1e-6, atol=1e-300)


def test_resume_is_exact(tmp_path):
    """A run killed at a checkpoint and restarted ends exactly where an uninterrupted one does."""
    from matmulnet.train import train
    kw = dict(seed=3, degrees=(-1, 1), max_epochs=40, ckpt_every=20, log_every=5, quiet=True)
    a = train(2, 7, str(tmp_path / "a"), **kw)
    train(2, 7, str(tmp_path / "b"), _interrupt_at=33, **kw)
    b = train(2, 7, str(tmp_path / "b"), **kw)
    assert a["val"] == b["val"] and a["epoch"] == b["epoch"]
    za, zb = np.load(tmp_path / "a" / "ckpt.npz"), np.load(tmp_path / "b" / "ckpt.npz")
    assert all(np.array_equal(za[k], zb[k]) for k in ("p0", "p1", "p2", "m0", "v0"))
    assert (tmp_path / "a" / "per_epoch.csv").read_text() == (tmp_path / "b" / "per_epoch.csv").read_text()
