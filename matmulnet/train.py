"""Train MatMulNet on n x n matrix multiplication with r multiplications.

    python -m matmulnet.train --n 3 --rank 23 --seed 1 --out runs/rank_n3_r23_seed1
    python -m matmulnet.train --n 3 --rank 19 --degrees -2 4 --seed 1 --out runs/border_r19_d-2_4_seed1

Protocol: training set of 10,000 and validation set of 1,000 triples (A, B, AB), entries uniform
in [-1, 1], drawn once; parameters initialised from N(0, 1) (see --init); Adam on the per-entry mean squared
error with mini-batches of 32, the training set being reshuffled at every epoch; gradient norm
clipped at 10; learning rate 1e-4, halved on plateaus of the validation loss (patience 15) down
to 1e-6. With --degrees the three factors are Laurent polynomials in epsilon, which decays
exponentially from 1e-1 to 1e-3 over the run. Everything is in double precision. The run reports
the best validation loss and stores the corresponding parameters.

Files written in --out: per_epoch.csv (epoch, epsilon, lr, training and validation loss),
best.npz (best parameters), summary.json, ckpt.npz (full state; an interrupted run restarts
from it and continues identically).
"""
import argparse
import json
import os
import time

import numpy as np

from . import metrics
from .data import draw_sets
from .schedule import EPS_START, Plateau, epsilon_at

MAX_EPOCHS, BATCH, CLIP, ALPHA = 1_000_000, 32, 10.0, 1.0


class NumbaBackend:
    def __init__(self, params, degrees, clip):
        from . import kernels
        self.k, self.clip = kernels, clip
        self.deg = np.asarray(degrees, dtype=np.float64)
        self.set_state(params, None, None, 0.0)

    def set_state(self, params, m, v, step):
        self.P = [np.ascontiguousarray(p, dtype=np.float64).copy() for p in params]
        self.M = [np.zeros_like(p) if m is None else m[i].copy() for i, p in enumerate(self.P)]
        self.V = [np.zeros_like(p) if v is None else v[i].copy() for i, p in enumerate(self.P)]
        self.step = float(step)

    def get_state(self):
        return ([p.copy() for p in self.P], [m.copy() for m in self.M],
                [v.copy() for v in self.V], self.step)

    def train_epoch(self, eps, lr, data, perm, batch):
        self.step, loss = self.k.train_epoch(*self.P, *self.M, *self.V, self.step,
                                             eps ** self.deg, lr, self.clip, *data, perm, batch)
        return loss

    def validate(self, eps, data):
        return float(self.k.mse(*self.P, eps ** self.deg, *data))


def make_backend(name, params, degrees, clip=CLIP):
    if name == "torch":
        from .reference_torch import TorchBackend
        return TorchBackend(params, degrees, clip)
    return NumbaBackend(params, degrees, clip)


def init_params(rng, n, r, degrees, init="normal", alpha=ALPHA, eps_start=EPS_START):
    """init="normal": every coefficient from N(0, alpha^2).
    init="scaled": the coefficient of eps^e from N(0, (alpha / eps_start^e)^2), so that every
    monomial eps^e * coefficient is of size alpha at the start of the epsilon schedule.
    The two coincide for the rank model (single exponent 0)."""
    m, d = n * n, len(degrees)
    params = [alpha * rng.standard_normal(s) for s in ((m, r, d), (m, r, d), (r, m, d))]
    if init == "scaled":
        scale = eps_start ** -np.asarray(degrees, dtype=np.float64)
        params = [p * scale for p in params]
    return params


def _save_ckpt(path, epoch, backend, sched, best, rng, data, cfg):
    P, M, V, step = backend.get_state()
    meta = dict(cfg, epoch=epoch, step=step, lr=sched.lr, pl_best=sched.best, pl_bad=sched.bad,
                best_epoch=best["epoch"], best_val=best["val"], rng_state=rng.bit_generator.state)
    arrays = {}
    for tag, group in (("p", P), ("m", M), ("v", V), ("b", best["P"])):
        arrays.update({f"{tag}{i}": x for i, x in enumerate(group)})
    arrays.update({f"train{i}": x for i, x in enumerate(data[0])})
    arrays.update({f"val{i}": x for i, x in enumerate(data[1])})
    tmp = path + ".tmp.npz"
    np.savez(tmp, meta=json.dumps(meta), **arrays)
    os.replace(tmp, path)


def _save_best(out, best, cfg):
    n, deg = cfg["n"], cfg["degrees"]
    H, K, F = (metrics.evaluate(p, deg, best["eps"]) for p in best["P"])
    np.savez(os.path.join(out, "best.npz"), H_coeff=best["P"][0], K_coeff=best["P"][1],
             F_coeff=best["P"][2], degrees=np.asarray(deg), epsilon=best["eps"], H=H, K=K, F=F,
             epoch=best["epoch"], val_loss=best["val"])
    summary = dict(cfg, best_val_loss=best["val"], best_epoch=best["epoch"],
                   epsilon_at_best=best["eps"], rho=metrics.rho(H, K, F, n),
                   weight=metrics.weight(H, K, F), epochs_done=best["done"])
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)


def train(n, rank, out, seed=0, degrees=None, max_epochs=MAX_EPOCHS, backend="numba", init="normal",
          log_every=100, ckpt_every=10_000, quiet=False, _interrupt_at=None):
    border = degrees is not None
    deg = list(range(degrees[0], degrees[1] + 1)) if border else [0]
    cfg = dict(n=n, rank=rank, degrees=deg, border=border, seed=seed, init=init, max_epochs=max_epochs,
               backend=backend)
    os.makedirs(out, exist_ok=True)
    ckpt = os.path.join(out, "ckpt.npz")
    eps_of = (lambda e: epsilon_at(e, max_epochs)) if border else (lambda e: 1.0)

    if os.path.exists(ckpt):
        z = np.load(ckpt)
        meta = json.loads(str(z["meta"]))
        for key in ("n", "rank", "degrees", "seed", "init", "max_epochs"):
            if meta[key] != cfg[key]:
                raise SystemExit(f"{ckpt} was written with {key}={meta[key]}, not {cfg[key]}")
        data = (tuple(z[f"train{i}"] for i in range(3)), tuple(z[f"val{i}"] for i in range(3)))
        rng = np.random.default_rng()
        rng.bit_generator.state = meta["rng_state"]
        be = make_backend(backend, [z[f"p{i}"] for i in range(3)], deg)
        be.set_state([z[f"p{i}"] for i in range(3)], [z[f"m{i}"] for i in range(3)],
                     [z[f"v{i}"] for i in range(3)], meta["step"])
        sched = Plateau(lr=meta["lr"], best=meta["pl_best"], bad=meta["pl_bad"])
        best = dict(P=[z[f"b{i}"].copy() for i in range(3)], val=meta["best_val"],
                    epoch=meta["best_epoch"], eps=eps_of(max(meta["best_epoch"], 1)))
        first = meta["epoch"] + 1
    else:
        rng = np.random.default_rng(seed)
        data = draw_sets(rng, n)
        params = init_params(rng, n, rank, deg, init)
        be = make_backend(backend, params, deg)
        sched = Plateau()
        best = dict(P=[p.copy() for p in params], val=float("inf"), epoch=0, eps=eps_of(1))
        first = 1
    train_set, val_set = data
    n_train = train_set[0].shape[0]

    log_path = os.path.join(out, "per_epoch.csv")
    kept = ["epoch,epsilon,lr,train_loss,val_loss\n"]
    if first > 1 and os.path.exists(log_path):      # drop the rows written after the checkpoint
        with open(log_path) as fh:
            kept += [row for row in fh.readlines()[1:] if int(row.split(",")[0]) < first]
    log = open(log_path, "w", buffering=1 << 20)
    log.writelines(kept)
    t0 = time.time()
    for epoch in range(first, max_epochs + 1):
        eps, lr = eps_of(epoch), sched.lr
        perm = rng.permutation(n_train)
        train_loss = be.train_epoch(eps, lr, train_set, perm, BATCH)
        val = be.validate(eps, val_set)
        sched.step(val)
        if val < best["val"]:
            best.update(P=be.get_state()[0], val=val, epoch=epoch, eps=eps)
        last = epoch == max_epochs
        if epoch % log_every == 0 or last:
            log.write(f"{epoch},{eps!r},{lr!r},{train_loss!r},{val!r}\n")
        if epoch % ckpt_every == 0 or last:
            log.flush()
            best["done"] = epoch
            _save_ckpt(ckpt, epoch, be, sched, best, rng, data, cfg)
            _save_best(out, best, cfg)
            if not quiet:
                rate = (time.time() - t0) / (epoch - first + 1)
                print(f"[{os.path.basename(out)}] epoch {epoch} lr {lr:.1e} val {val:.4e} "
                      f"best {best['val']:.4e} @ {best['epoch']} | {rate * 1e3:.2f} ms/epoch, "
                      f"{(max_epochs - epoch) * rate / 3600:.2f} h left", flush=True)
        if epoch == _interrupt_at:      # used by the tests to emulate a killed process
            break
    log.close()
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=3, help="matrix size")
    ap.add_argument("--rank", type=int, required=True, help="number of multiplications r")
    ap.add_argument("--degrees", type=int, nargs=2, metavar=("D_MIN", "D_MAX"),
                    help="exponent range of the epsilon-parametrised (border rank) model")
    ap.add_argument("--init", choices=["normal", "scaled"], default="normal",
                    help="initial coefficients: N(0, 1) (default), or N(0, 1) divided by "
                         "eps_start^e for the coefficient of eps^e")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--out", required=True, help="run directory")
    ap.add_argument("--backend", choices=["numba", "torch"], default="numba")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=10_000)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    if a.degrees and a.degrees[0] > a.degrees[1]:
        ap.error("--degrees needs D_MIN <= D_MAX")
    best = train(a.n, a.rank, a.out, a.seed, a.degrees, a.max_epochs, a.backend, a.init, a.log_every,
                 a.ckpt_every, a.quiet)
    print(f"best validation loss {best['val']:.6e} at epoch {best['epoch']}")


if __name__ == "__main__":
    main()
