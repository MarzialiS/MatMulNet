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

Files written in --out: per_epoch.csv (epoch, epsilon, lr, training and validation loss, and
the full-tensor residual rho and the weight W of the current factors), best.npz and last.npz
(parameters at the best validation loss and at the last epoch), summary.json, ckpt.npz (full
state; an interrupted run restarts from it and continues identically).

--init-from FILE starts from the factors stored in FILE (a best.npz, or a known scheme written
by experiments/known_schemes.py) instead of a random point. --wandb ENTITY/PROJECT also sends
the logged quantities to Weights & Biases.

Departures from the protocol, all off by default: --lr, --lr-min, --lr-factor and --patience
change the plateau schedule; --schedule cosine decays the learning rate from --lr to --lr-min
over the run, --schedule constant keeps it at --lr; --batch changes the size of the mini-batches;
--weight-decay WD uses AdamW, multiplying the parameters by 1 - lr * WD at every update.

Over-parametrisation and pruning: a --init-from file with fewer terms than --rank is completed
with new terms whose coefficients are drawn from N(0, --extra-scale^2); --group-lasso LAM adds,
after every epoch, the proximal step of the penalty lam * sum_s ||theta_s|| (theta_s: all the
coefficients of term s), with lam growing linearly from 0 to LAM over the run, which drives the
terms that are not needed to zero; --prune-to R makes the polish start from the R terms of
largest weight ||h_s|| ||k_s|| ||f_s|| of the best parameters.
--polish N, for the rank model, ends the run with at most N iterations of unconstrained
Levenberg-Marquardt on the full-tensor residual from the best parameters (refinement.refine),
and records the outcome in summary.json (polish_rho, polish_weight, ...) and in polished.npz.
"""
import argparse
import json
import os
import time
import uuid

import numpy as np

from . import metrics
from .data import draw_sets
from .schedule import EPS_START, LR_FACTOR, LR_MIN, LR_PATIENCE, LR_START, Plateau, cosine_lr, \
    epsilon_at

MAX_EPOCHS, BATCH, CLIP, ALPHA = 1_000_000, 32, 10.0, 1.0


class NumbaBackend:
    def __init__(self, params, degrees, clip, weight_decay=0.0):
        from . import kernels
        self.k, self.clip, self.wd = kernels, clip, weight_decay
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
                                             eps ** self.deg, lr, self.clip, *data, perm, batch,
                                             1.0 - lr * self.wd)
        return loss

    def validate(self, eps, data):
        return float(self.k.mse(*self.P, eps ** self.deg, *data))


def make_backend(name, params, degrees, clip=CLIP, weight_decay=0.0):
    if name == "torch":
        from .reference_torch import TorchBackend
        return TorchBackend(params, degrees, clip, weight_decay)
    return NumbaBackend(params, degrees, clip, weight_decay)


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


def _save_factors(path, P, cfg, eps, epoch, val):
    """Write coefficients and factors; return (rho, W) of the factors at this epsilon."""
    n, deg = cfg["n"], cfg["degrees"]
    H, K, F = (metrics.evaluate(p, deg, eps) for p in P)
    np.savez(path, H_coeff=P[0], K_coeff=P[1], F_coeff=P[2], degrees=np.asarray(deg), epsilon=eps,
             H=H, K=K, F=F, epoch=epoch, val_loss=val, n=n, dtype="float64")
    return metrics.rho(H, K, F, n), metrics.weight(H, K, F)


def _save_results(out, best, last, cfg):
    rho, w = _save_factors(os.path.join(out, "best.npz"), best["P"], cfg, best["eps"],
                           best["epoch"], best["val"])
    rho_l, w_l = _save_factors(os.path.join(out, "last.npz"), last["P"], cfg, last["eps"],
                               last["epoch"], last["val"])
    summary = dict(cfg, precision="float64", best_val_loss=best["val"], best_epoch=best["epoch"],
                   epsilon_at_best=best["eps"], rho=rho, weight=w, last_val_loss=last["val"],
                   epsilon_at_last=last["eps"], last_rho=rho_l, last_weight=w_l,
                   epochs_done=last["epoch"], updates=int(last["step"]),
                   seconds=last["seconds"])
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def _load_start(path, n, rank, deg, rng=None, extra_scale=0.0):
    """Coefficient tensors from a file holding either H_coeff, K_coeff, F_coeff and degrees
    (possibly a narrower exponent range than `deg`) or plain factors H, K, F. A file with fewer
    terms than `rank` is completed with terms drawn from N(0, extra_scale^2)."""
    z = np.load(path)
    m = n * n
    if "H_coeff" in z.files:
        src = [int(d) for d in z["degrees"]]
        coeff = [np.asarray(z[k], dtype=np.float64) for k in ("H_coeff", "K_coeff", "F_coeff")]
    else:
        src = [0]
        coeff = [np.asarray(z[k], dtype=np.float64)[..., None] for k in ("H", "K", "F")]
    if not set(src) <= set(deg):
        raise SystemExit(f"{path} has exponents {src}, the model has {deg}")
    r0 = coeff[0].shape[1]
    if r0 > rank or coeff[0].shape[0] != m:
        raise SystemExit(f"{path}: {r0} terms of size {coeff[0].shape[0]}, the model has {rank} "
                         f"of size {m}")
    params = []
    for j, (c, shape) in enumerate(zip(coeff, ((m, rank), (m, rank), (rank, m)))):
        p = np.zeros(shape + (len(deg),))
        for i, d in enumerate(src):
            if j < 2:
                p[:, :r0, deg.index(d)] = c[..., i]
            else:
                p[:r0, :, deg.index(d)] = c[..., i]
        params.append(p)
    if r0 < rank:
        for j, p in enumerate(params):
            new = extra_scale * rng.standard_normal((m, rank - r0, len(deg)) if j < 2 else
                                                    (rank - r0, m, len(deg)))
            if j < 2:
                p[:, r0:, :] = new
            else:
                p[r0:, :, :] = new
    return params


def _group_shrink(P, amount):
    """Proximal step of amount * sum_s ||theta_s||: every term is shrunk towards zero by
    `amount` in norm, and set to zero if it is smaller."""
    H, K, F = P
    norms = np.sqrt((H ** 2).sum(axis=(0, 2)) + (K ** 2).sum(axis=(0, 2)) + (F ** 2).sum(axis=(1, 2)))
    f = np.maximum(0.0, 1.0 - amount / np.maximum(norms, 1e-300))
    return [H * f[None, :, None], K * f[None, :, None], F * f[:, None, None]]


def _polish(out, best, cfg, max_iter):
    """Unconstrained Levenberg-Marquardt from the best parameters (rank model), after keeping
    the cfg["prune_to"] terms of largest weight if that is set."""
    from refinement.refine import Problem, projected_lm
    n = cfg["n"]
    H, K, F = (p[..., 0] for p in best["P"])
    pruned = {}
    if cfg["prune_to"] and cfg["prune_to"] < H.shape[1]:
        w = np.linalg.norm(H, axis=0) * np.linalg.norm(K, axis=0) * np.linalg.norm(F, axis=1)
        keep = np.sort(np.argsort(w)[::-1][:cfg["prune_to"]])
        dropped = np.setdiff1d(np.arange(H.shape[1]), keep)
        pruned = dict(polish_dropped_weight=float(w[dropped].sum()),
                      polish_rho_after_pruning=metrics.rho(H[:, keep], K[:, keep], F[keep], n))
        H, K, F = H[:, keep], K[:, keep], F[keep]
    prob = Problem(n, H.shape[1])
    t0 = time.time()
    x, status, iters = projected_lm(prob, prob.pack(H, K, F), np.inf, max_iter,
                                    stall_window=max_iter)
    H, K, F = prob.unpack(x)
    res = dict(polish_rho=metrics.rho(H, K, F, n), polish_weight=metrics.weight(H, K, F),
               polish_status=status, polish_iterations=iters,
               polish_cancellation_level=metrics.cancellation_level(H, K, F, n),
               polish_seconds=time.time() - t0, polish_rank=H.shape[1], **pruned)
    np.savez(os.path.join(out, "polished.npz"), H=H, K=K, F=F, n=n, **res)
    return res


def _wandb_run(target, out, cfg, group, tags, job_type=None):
    import wandb
    entity, _, project = target.rpartition("/")
    id_file = os.path.join(out, "wandb_id.txt")
    run_id = open(id_file).read().strip() if os.path.exists(id_file) else uuid.uuid4().hex[:8]
    with open(id_file, "w") as fh:
        fh.write(run_id)
    kind = "border" if cfg["border"] else "rank"
    return wandb.init(entity=entity or None, project=project, id=run_id, resume="allow",
                      name=os.path.basename(os.path.normpath(out)), config=cfg,
                      group=group or f"{kind}_n{cfg['n']}_r{cfg['rank']}",
                      job_type=job_type or ("control" if cfg["init_from"] else kind),
                      tags=tags or None,
                      settings=wandb.Settings(console="off"))


def train(n, rank, out, seed=0, degrees=None, max_epochs=MAX_EPOCHS, backend="numba", init="normal",
          log_every=100, ckpt_every=10_000, quiet=False, init_from=None, wandb=None,
          wandb_group=None, wandb_tags=None, lr=LR_START, lr_min=LR_MIN, lr_factor=LR_FACTOR,
          patience=LR_PATIENCE, schedule="plateau", batch=BATCH, polish=0, wandb_job_type=None,
          weight_decay=0.0, extra_scale=0.1, group_lasso=0.0, prune_to=0, _interrupt_at=None):
    border = degrees is not None
    deg = list(range(degrees[0], degrees[1] + 1)) if border else [0]
    cfg = dict(n=n, rank=rank, degrees=deg, border=border, seed=seed, init=init, max_epochs=max_epochs,
               backend=backend, init_from=init_from, lr_start=lr, lr_min=lr_min, lr_factor=lr_factor,
               patience=patience, schedule=schedule, batch=batch, polish=polish,
               weight_decay=weight_decay, extra_scale=extra_scale, group_lasso=group_lasso,
               prune_to=prune_to)
    # the protocol's values, for checkpoints written before these options existed
    old = dict(lr_start=LR_START, lr_min=LR_MIN, lr_factor=LR_FACTOR, patience=LR_PATIENCE,
               schedule="plateau", batch=BATCH, weight_decay=0.0, extra_scale=0.1,
               group_lasso=0.0)
    os.makedirs(out, exist_ok=True)
    ckpt = os.path.join(out, "ckpt.npz")
    eps_of = (lambda e: epsilon_at(e, max_epochs)) if border else (lambda e: 1.0)

    if os.path.exists(ckpt):
        z = np.load(ckpt)
        meta = json.loads(str(z["meta"]))
        for key in ("n", "rank", "degrees", "seed", "init", "max_epochs", "init_from", *old):
            if meta.get(key, old.get(key)) != cfg[key]:
                raise SystemExit(f"{ckpt} was written with {key}={meta.get(key, old.get(key))}, "
                                 f"not {cfg[key]}")
        data = (tuple(z[f"train{i}"] for i in range(3)), tuple(z[f"val{i}"] for i in range(3)))
        rng = np.random.default_rng()
        rng.bit_generator.state = meta["rng_state"]
        be = make_backend(backend, [z[f"p{i}"] for i in range(3)], deg, CLIP, weight_decay)
        be.set_state([z[f"p{i}"] for i in range(3)], [z[f"m{i}"] for i in range(3)],
                     [z[f"v{i}"] for i in range(3)], meta["step"])
        sched = Plateau(lr=meta["lr"], factor=lr_factor, patience=patience, min_lr=lr_min,
                        best=meta["pl_best"], bad=meta["pl_bad"])
        best = dict(P=[z[f"b{i}"].copy() for i in range(3)], val=meta["best_val"],
                    epoch=meta["best_epoch"], eps=eps_of(max(meta["best_epoch"], 1)))
        first, elapsed = meta["epoch"] + 1, meta.get("seconds", 0.0)
    else:
        rng = np.random.default_rng(seed)
        data = draw_sets(rng, n)
        params = _load_start(init_from, n, rank, deg, rng, extra_scale) if init_from else \
            init_params(rng, n, rank, deg, init)
        be = make_backend(backend, params, deg, CLIP, weight_decay)
        sched = Plateau(lr=lr, factor=lr_factor, patience=patience, min_lr=lr_min)
        best = dict(P=[p.copy() for p in params], val=float("inf"), epoch=0, eps=eps_of(1))
        first, elapsed = 1, 0.0
    train_set, val_set = data
    n_train = train_set[0].shape[0]

    log_path = os.path.join(out, "per_epoch.csv")
    kept = ["epoch,epsilon,lr,train_loss,val_loss,rho,weight\n"]
    if first > 1 and os.path.exists(log_path):      # drop the rows written after the checkpoint
        with open(log_path) as fh:
            kept += [row for row in fh.readlines()[1:] if int(row.split(",")[0]) < first]
    log = open(log_path, "w", buffering=1 << 20)
    log.writelines(kept)
    run = _wandb_run(wandb, out, cfg, wandb_group, wandb_tags, wandb_job_type) if wandb else None
    lr_of = {"plateau": lambda e: sched.lr, "constant": lambda e: lr,
             "cosine": lambda e: cosine_lr(e, max_epochs, lr, lr_min)}[schedule]
    t0 = time.time()
    for epoch in range(first, max_epochs + 1):
        eps, lr_now = eps_of(epoch), lr_of(epoch)
        perm = rng.permutation(n_train)
        train_loss = be.train_epoch(eps, lr_now, train_set, perm, batch)
        val = be.validate(eps, val_set)
        if schedule == "plateau":
            sched.step(val)
        if group_lasso:
            P_now, M_now, V_now, step = be.get_state()
            amount = lr_now * group_lasso * epoch / max_epochs * -(-n_train // batch)
            be.set_state(_group_shrink(P_now, amount), M_now, V_now, step)
            val = be.validate(eps, val_set)
        if val < best["val"]:
            best.update(P=be.get_state()[0], val=val, epoch=epoch, eps=eps)
        last = epoch == max_epochs
        if epoch % log_every == 0 or last:
            P_now, _, _, step = be.get_state()
            H, K, F = (metrics.evaluate(p, deg, eps) for p in P_now)
            rho, w = metrics.rho(H, K, F, n), metrics.weight(H, K, F)
            log.write(f"{epoch},{eps!r},{lr_now!r},{train_loss!r},{val!r},{rho!r},{w!r}\n")
            if run:
                run.log(dict(epoch=epoch, epsilon=eps, lr=lr_now, train_loss=train_loss, val_loss=val,
                             rho=rho, weight=w, best_val_loss=best["val"],
                             updates=int(step)), step=epoch)
        if epoch % ckpt_every == 0 or last:
            log.flush()
            seconds = elapsed + time.time() - t0
            _save_ckpt(ckpt, epoch, be, sched, best, rng, data, dict(cfg, seconds=seconds))
            P_now, _, _, step = be.get_state()
            summary = _save_results(out, best, dict(P=P_now, eps=eps, epoch=epoch, val=val,
                                                    step=step, seconds=seconds), cfg)
            if run:
                run.summary.update({k: v for k, v in summary.items() if k not in cfg})
            if not quiet:
                rate = (time.time() - t0) / (epoch - first + 1)
                print(f"[{os.path.basename(out)}] epoch {epoch} lr {lr_now:.1e} val {val:.4e} "
                      f"best {best['val']:.4e} @ {best['epoch']} | {rate * 1e3:.2f} ms/epoch, "
                      f"{(max_epochs - epoch) * rate / 3600:.2f} h left", flush=True)
        if epoch == _interrupt_at:      # used by the tests to emulate a killed process
            break
    log.close()
    summary_path = os.path.join(out, "summary.json")
    if polish and not border and os.path.exists(summary_path):
        with open(summary_path) as fh:
            summary = json.load(fh)
        if summary["epochs_done"] >= max_epochs and "polish_rho" not in summary:
            # with the group lasso the terms are pruned along the run: polish the last point
            start = dict(P=be.get_state()[0]) if group_lasso else best
            summary.update(_polish(out, start, cfg, polish))
            with open(summary_path, "w") as fh:
                json.dump(summary, fh, indent=2)
            if run:
                run.summary.update({k: v for k, v in summary.items() if k.startswith("polish_")})
            if not quiet:
                print(f"[{os.path.basename(out)}] polish: rho {summary['polish_rho']:.3e} "
                      f"W {summary['polish_weight']:.4g} {summary['polish_status']} "
                      f"({summary['polish_iterations']} it)", flush=True)
    if run:
        run.finish()
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
    ap.add_argument("--init-from", metavar="FILE", help="start from the factors stored in FILE")
    ap.add_argument("--wandb", metavar="ENTITY/PROJECT", help="log to Weights & Biases")
    ap.add_argument("--wandb-group")
    ap.add_argument("--wandb-tags", nargs="*")
    ap.add_argument("--wandb-job-type")
    ap.add_argument("--lr", type=float, default=LR_START, help="initial learning rate")
    ap.add_argument("--lr-min", type=float, default=LR_MIN, help="smallest learning rate")
    ap.add_argument("--lr-factor", type=float, default=LR_FACTOR, help="plateau: reduction factor")
    ap.add_argument("--patience", type=int, default=LR_PATIENCE, help="plateau: patience (epochs)")
    ap.add_argument("--schedule", choices=["plateau", "cosine", "constant"], default="plateau")
    ap.add_argument("--batch", type=int, default=BATCH, help="mini-batch size")
    ap.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay")
    ap.add_argument("--extra-scale", type=float, default=0.1,
                    help="scale of the terms added to a --init-from file with fewer terms")
    ap.add_argument("--group-lasso", type=float, default=0.0, metavar="LAM",
                    help="group-lasso weight on the terms, growing linearly to LAM")
    ap.add_argument("--prune-to", type=int, default=0, metavar="R",
                    help="polish from the R terms of largest weight")
    ap.add_argument("--polish", type=int, default=0, metavar="N",
                    help="end with at most N Levenberg-Marquardt iterations (rank model)")
    a = ap.parse_args()
    if a.degrees and a.degrees[0] > a.degrees[1]:
        ap.error("--degrees needs D_MIN <= D_MAX")
    best = train(a.n, a.rank, a.out, a.seed, a.degrees, a.max_epochs, a.backend, a.init, a.log_every,
                 a.ckpt_every, a.quiet, a.init_from, a.wandb, a.wandb_group, a.wandb_tags, a.lr,
                 a.lr_min, a.lr_factor, a.patience, a.schedule, a.batch, a.polish, a.wandb_job_type,
                 a.weight_decay, a.extra_scale, a.group_lasso, a.prune_to)
    print(f"best validation loss {best['val']:.6e} at epoch {best['epoch']}")


if __name__ == "__main__":
    main()
