"""Train a rank-r decomposition of M_n on the exact loss, computed on the reconstructed tensor.

    python -m matmulnet.exact --n 4 --rank 48 --init-from start.npz --out runs/x --lr 1e-3

The loss is the per-entry mean squared error that the sampled protocol of matmulnet.train
estimates, in closed form: with A, B having independent entries uniform in [-1, 1] (variance
1/3), E ||vec(AB) - F^T((H^T vec A) * (K^T vec B))||^2 / n^2 = ||T - M_n||_F^2 / (9 n^2), with T
the tensor of the factors. So

    loss = ||T - M_n||_F^2 / (9 n^2) = rho^2 n / 9          (4 rho^2 / 9 for n = 4),

learning rates and the gradient clipping keep the meaning they have in the sampled protocol, and
there are no training or validation sets: every step is a full-batch Adam step on the exact
loss. The run is deterministic once the starting point is fixed.

The learning rate follows --schedule: plateau (the rule of torch's ReduceLROnPlateau on the loss,
evaluated every --chunk steps, but reaching --lr-min however small: see schedule.Plateau;
--patience is counted in steps), cosine (from --lr to --lr-min
over --steps) or constant. The run stops after --steps steps, when rho falls below --stop-rho,
or when the best loss has not improved by a relative 1e-3 for --stall steps while the learning
rate cannot decrease any more (plateau: it is at --lr-min; the other schedules: always).
--polish N ends with at most N Levenberg-Marquardt iterations from the best point, as in
matmulnet.train.

Pressures towards discrete solutions (the coefficients of the known schemes lie on a small
grid: 0, +-1/8, +-1/4, +-1/2, +-1 for the rational 48-term scheme): --quant LAM adds the penalty
lam * sum_i dist(theta_i, grid)^2, lam growing linearly from 0 to LAM over the run; --l1 LAM
the L1 proximal step (soft threshold lr * lam, lam decreasing from LAM to 0); --balance rescales
every term to equal norms of its three factors; --freeze-every N snaps to the grid, and
freezes, every N steps the entries within delta of it (delta from --freeze-delta, after the
fraction --freeze-start of the run). With any of them the run takes the whole budget (or stops
when the rounded point is exact), the polish starts from the last point, and rounded.npz holds
the point rounded to the grid.

Files written in --out: per_step.csv (step, lr, loss, rho, W every --log-every steps), best.npz
and last.npz (format of matmulnet.train), summary.json, ckpt.npz (complete state: a run that
is interrupted continues identically), polished.npz with --polish.

--extend continues a run from its checkpoint with other --steps, --stall and --lr-min, also
after it has stopped on the budget or on a stall (its polish is then done again at the end).
"""
import argparse
import json
import os
import time

import numpy as np
from numba import njit

from . import metrics
from .kernels import ADAM_EPS, BETA1, BETA2, _adam
from .schedule import Plateau, cosine_lr
from .train import _load_start, _polish, _wandb_run

CLIP = 10.0
GRID = (0.0, 0.125, 0.25, 0.5, 1.0)       # with signs: the coefficients of the rational 48-term scheme


@njit(cache=True, fastmath=True, boundscheck=False, nogil=True, error_model="numpy")
def loss_grad(H, K, F, M, gH, gK, gF):
    """Exact loss ||T - M||^2 / (9 m), m = n^2, and its gradient (written into gH, gK, gF).
    One pass over the pairs (a, b): w = H[a] * K[b], e = w F - M[a, b], and
    dF += w x e, dH[a] += K[b] * (F e), dK[b] += H[a] * (F e)."""
    m, r = H.shape
    sc = 2.0 / (9.0 * m)
    w = np.empty(r)
    e = np.empty(m)
    for s in range(r):
        for c in range(m):
            gF[s, c] = 0.0
    for a in range(m):
        for s in range(r):
            gH[a, s] = 0.0
            gK[a, s] = 0.0
    tot = 0.0
    for a in range(m):
        for b in range(m):
            for s in range(r):
                w[s] = H[a, s] * K[b, s]
            for c in range(m):
                e[c] = -M[a, b, c]
            for s in range(r):
                ws = w[s]
                for c in range(m):
                    e[c] += ws * F[s, c]
            for c in range(m):
                tot += e[c] * e[c]
                e[c] *= sc
            for s in range(r):
                ws = w[s]
                t = 0.0
                for c in range(m):
                    gF[s, c] += ws * e[c]
                    t += F[s, c] * e[c]
                gH[a, s] += K[b, s] * t
                gK[b, s] += H[a, s] * t
    return tot / (9.0 * m)


@njit(cache=True, fastmath=True, boundscheck=False, nogil=True, error_model="numpy")
def adam_steps(H, K, F, mH, mK, mF, vH, vK, vF, step, lr, clip, M, losses):
    """len(losses) Adam steps on the exact loss, with the gradient norm clipped at `clip` (as
    torch.nn.utils.clip_grad_norm_). losses[i] is the loss before step i. Returns the step
    counter."""
    m, r = H.shape
    gH = np.empty((m, r))
    gK = np.empty((m, r))
    gF = np.empty((r, m))
    gHf, gKf, gFf = gH.reshape(-1), gK.reshape(-1), gF.reshape(-1)
    for it in range(losses.shape[0]):
        losses[it] = loss_grad(H, K, F, M, gH, gK, gF)
        g2 = 0.0
        for i in range(gHf.shape[0]):
            g2 += gHf[i] * gHf[i] + gKf[i] * gKf[i] + gFf[i] * gFf[i]
        coef = clip / (np.sqrt(g2) + 1e-6)
        if coef < 1.0:
            for i in range(gHf.shape[0]):
                gHf[i] *= coef
                gKf[i] *= coef
                gFf[i] *= coef
        step += 1.0
        step_size = lr / (1.0 - BETA1 ** step)
        inv_bc2_sqrt = 1.0 / np.sqrt(1.0 - BETA2 ** step)
        _adam(H.reshape(-1), mH.reshape(-1), vH.reshape(-1), gHf, step_size, inv_bc2_sqrt, 1.0)
        _adam(K.reshape(-1), mK.reshape(-1), vK.reshape(-1), gKf, step_size, inv_bc2_sqrt, 1.0)
        _adam(F.reshape(-1), mF.reshape(-1), vF.reshape(-1), gFf, step_size, inv_bc2_sqrt, 1.0)
    return step


@njit(cache=True, fastmath=True, boundscheck=False, nogil=True, error_model="numpy")
def _nearest(x, grid):
    """The point of the symmetric grid {+-g : g in grid} nearest to x."""
    a = abs(x)
    best = grid[0]
    for g in grid:
        if abs(a - g) < abs(a - best):
            best = g
    return best if x >= 0.0 else -best


@njit(cache=True, fastmath=True, boundscheck=False, nogil=True, error_model="numpy")
def _adam_discrete(p, m, v, g, free, step_size, inv_bc2_sqrt, quant, grid, shrink):
    """Adam on the free entries (free[i] = 1; the others keep their value) with the gradient
    of quant * dist(p, grid)^2 added to g, then the soft threshold of the L1 prox (shrink)."""
    for i in range(p.size):
        if free[i] == 0.0:
            continue
        pi = p[i]
        gi = g[i] + 2.0 * quant * (pi - _nearest(pi, grid))
        mi = BETA1 * m[i] + (1.0 - BETA1) * gi
        vi = BETA2 * v[i] + (1.0 - BETA2) * gi * gi
        m[i] = mi
        v[i] = vi
        pi -= step_size * mi / (np.sqrt(vi) * inv_bc2_sqrt + ADAM_EPS)
        if shrink > 0.0:
            if pi > shrink:
                pi -= shrink
            elif pi < -shrink:
                pi += shrink
            else:
                pi = 0.0
        p[i] = pi


@njit(cache=True, fastmath=True, boundscheck=False, nogil=True, error_model="numpy")
def adam_steps_discrete(H, K, F, mH, mK, mF, vH, vK, vF, step, lr, clip, M, losses, quant, l1,
                        grid, fH, fK, fF):
    """adam_steps with the pressures towards discrete solutions: the penalty
    quant * sum_i dist(theta_i, grid)^2 (not clipped, not counted in losses), the L1 proximal
    step of weight l1 (threshold lr * l1), and frozen entries (f = 0)."""
    m, r = H.shape
    gH = np.empty((m, r))
    gK = np.empty((m, r))
    gF = np.empty((r, m))
    gHf, gKf, gFf = gH.reshape(-1), gK.reshape(-1), gF.reshape(-1)
    fHf, fKf, fFf = fH.reshape(-1), fK.reshape(-1), fF.reshape(-1)
    for it in range(losses.shape[0]):
        losses[it] = loss_grad(H, K, F, M, gH, gK, gF)
        g2 = 0.0
        for i in range(gHf.shape[0]):
            g2 += fHf[i] * gHf[i] * gHf[i] + fKf[i] * gKf[i] * gKf[i] + fFf[i] * gFf[i] * gFf[i]
        coef = clip / (np.sqrt(g2) + 1e-6)
        if coef < 1.0:
            for i in range(gHf.shape[0]):
                gHf[i] *= coef
                gKf[i] *= coef
                gFf[i] *= coef
        step += 1.0
        step_size = lr / (1.0 - BETA1 ** step)
        inv_bc2_sqrt = 1.0 / np.sqrt(1.0 - BETA2 ** step)
        shrink = lr * l1
        _adam_discrete(H.reshape(-1), mH.reshape(-1), vH.reshape(-1), gHf, fHf, step_size,
                       inv_bc2_sqrt, quant, grid, shrink)
        _adam_discrete(K.reshape(-1), mK.reshape(-1), vK.reshape(-1), gKf, fKf, step_size,
                       inv_bc2_sqrt, quant, grid, shrink)
        _adam_discrete(F.reshape(-1), mF.reshape(-1), vF.reshape(-1), gFf, fFf, step_size,
                       inv_bc2_sqrt, quant, grid, shrink)
    return step


def rounded(P, grid):
    """Every coefficient replaced by the nearest point of the symmetric grid."""
    g = np.concatenate([-grid[::-1], grid])
    return [g[np.argmin(np.abs(p[..., None] - g), axis=-1)] for p in P]


def balance(P):
    """Rescale every term so that ||h_s|| = ||k_s|| = ||f_s|| (the term is unchanged)."""
    H, K, F = P
    nh, nk, nf = np.linalg.norm(H, axis=0), np.linalg.norm(K, axis=0), np.linalg.norm(F, axis=1)
    ok = (nh > 0) & (nk > 0) & (nf > 0)
    g = np.cbrt(nh * nk * nf)
    for x, nx, ax in ((H, nh, 0), (K, nk, 0), (F, nf, 1)):
        f = np.where(ok, g / np.where(ok, nx, 1.0), 1.0)
        x *= f[None, :] if ax == 0 else f[:, None]


def exact_loss(H, K, F, M):
    return loss_grad(H, K, F, M, np.empty_like(H), np.empty_like(K), np.empty_like(F))


def rho_of_loss(loss, n):
    return float(np.sqrt(max(loss, 0.0) * 9.0 / n))


def _save_factors(path, P, n, step, loss):
    H, K, F = P
    np.savez(path, H_coeff=H[..., None], K_coeff=K[..., None], F_coeff=F[..., None],
             degrees=np.array([0]), epsilon=1.0, H=H, K=K, F=F, step=step, loss=loss, n=n,
             dtype="float64")
    return metrics.rho(H, K, F, n), metrics.weight(H, K, F)


def train_exact(n, rank, out, seed=0, init_from=None, steps=2_000_000, lr=1e-4, lr_min=1e-6,
                lr_factor=0.5, patience=10_000, schedule="plateau", clip=CLIP, chunk=100,
                stop_rho=1e-8, stall=100_000, polish=0, log_every=1000, ckpt_every=200_000,
                quiet=False, wandb=None, wandb_group=None, wandb_tags=None, wandb_job_type=None,
                quant=0.0, l1=0.0, grid=GRID, balance_terms=False, freeze_every=0,
                freeze_delta=(0.05, 0.05), freeze_start=0.0, extend=False, _interrupt_at=None):
    if steps % chunk or log_every % chunk or ckpt_every % chunk or freeze_every % chunk:
        raise SystemExit("--steps, --log-every, --ckpt-every and --freeze-every must be "
                         "multiples of --chunk")
    discrete = bool(quant or l1 or balance_terms or freeze_every)
    if balance_terms and freeze_every:
        raise SystemExit("--balance rescales the frozen entries: use one or the other")
    cfg = dict(n=n, rank=rank, seed=seed, init_from=init_from, loss="exact", steps=steps,
               lr_start=lr, lr_min=lr_min, lr_factor=lr_factor, patience=patience,
               schedule=schedule, clip=clip, chunk=chunk, stop_rho=stop_rho, stall=stall,
               polish=polish, border=False, prune_to=0, discrete=discrete, quant=quant, l1=l1,
               grid=[float(g) for g in grid], balance=balance_terms, freeze_every=freeze_every,
               freeze_delta=[float(d) for d in freeze_delta], freeze_start=freeze_start)
    # values of the options added after the first runs, for their checkpoints
    old = dict(discrete=False, quant=0.0, l1=0.0, grid=list(GRID), balance=False,
               freeze_every=0, freeze_delta=[0.05, 0.05], freeze_start=0.0)
    grid_arr = np.asarray(sorted(grid), dtype=np.float64)
    os.makedirs(out, exist_ok=True)
    ckpt = os.path.join(out, "ckpt.npz")
    M = metrics.matmul_tensor(n)
    n_chunks = steps // chunk
    sched_patience = max(1, patience // chunk)

    if os.path.exists(ckpt):
        z = np.load(ckpt)
        meta = json.loads(str(z["meta"]))
        for key in cfg:
            if extend and key in ("steps", "stall", "lr_min"):
                continue
            if meta.get(key, old.get(key)) != cfg[key]:
                raise SystemExit(f"{ckpt} was written with {key}={meta.get(key, old.get(key))}, "
                                 f"not {cfg[key]}")
        P, Mo, V = ([z[f"{t}{i}"].copy() for i in range(3)] for t in "pmv")
        free = [z[f"f{i}"].copy() for i in range(3)] if "f0" in z.files else \
            [np.ones_like(p) for p in P]
        best = dict(P=[z[f"b{i}"].copy() for i in range(3)], loss=meta["best_loss"],
                    step=meta["best_step"])
        sched = Plateau(lr=meta["lr"], factor=lr_factor, patience=sched_patience, min_lr=lr_min,
                        best=meta["pl_best"], bad=meta["pl_bad"])
        first, adam_step, elapsed = meta["chunk_done"] + 1, meta["adam_step"], meta["seconds"]
        last_gain, stop = meta["last_gain"], meta["stop"]
        if extend and stop in ("budget", "stall"):
            stop = None
    else:
        rng = np.random.default_rng(seed)
        if init_from:
            P = [np.ascontiguousarray(p[..., 0]) for p in _load_start(init_from, n, rank, [0])]
        else:
            m = n * n
            P = [rng.standard_normal(s) for s in ((m, rank), (m, rank), (rank, m))]
        Mo, V = [np.zeros_like(p) for p in P], [np.zeros_like(p) for p in P]
        free = [np.ones_like(p) for p in P]
        loss0 = exact_loss(*P, M)
        best = dict(P=[p.copy() for p in P], loss=loss0, step=0)
        sched = Plateau(lr=lr, factor=lr_factor, patience=sched_patience, min_lr=lr_min)
        first, adam_step, elapsed, last_gain, stop = 1, 0.0, 0.0, 0, None

    log_path = os.path.join(out, "per_step.csv")
    kept = ["step,lr,loss,rho,weight" + (",frozen,rho_rounded" if discrete else "") + "\n"]
    if first > 1 and os.path.exists(log_path):      # drop the rows written after the checkpoint
        with open(log_path) as fh:
            kept += [row for row in fh.readlines()[1:] if int(row.split(",")[0]) < first * chunk]
    log = open(log_path, "w", buffering=1 << 20)
    log.writelines(kept)
    run = _wandb_run(wandb, out, cfg, wandb_group, wandb_tags, wandb_job_type) if wandb else None
    lr_of = {"plateau": lambda c: sched.lr, "constant": lambda c: lr,
             "cosine": lambda c: cosine_lr(c, n_chunks, lr, lr_min)}[schedule]
    losses = np.empty(chunk)
    t0 = time.time()

    def save(c):
        log.flush()
        seconds = elapsed + time.time() - t0
        meta = dict(cfg, chunk_done=c, adam_step=adam_step, lr=sched.lr, pl_best=sched.best,
                    pl_bad=sched.bad, best_loss=best["loss"], best_step=best["step"],
                    seconds=seconds, last_gain=last_gain, stop=stop)
        groups = (("p", P), ("m", Mo), ("v", V), ("b", best["P"])) + \
            ((("f", free),) if discrete else ())
        arrays = {f"{t}{i}": x for t, g in groups for i, x in enumerate(g)}
        np.savez(ckpt + ".tmp.npz", meta=json.dumps(meta), **arrays)
        os.replace(ckpt + ".tmp.npz", ckpt)
        loss_now = exact_loss(*P, M)
        rho_b, w_b = _save_factors(os.path.join(out, "best.npz"), best["P"], n, best["step"],
                                   best["loss"])
        rho_l, w_l = _save_factors(os.path.join(out, "last.npz"), P, n, c * chunk, loss_now)
        summary = dict(cfg, precision="float64", best_loss=best["loss"], best_step=best["step"],
                       rho=rho_b, weight=w_b, last_loss=loss_now, last_rho=rho_l, last_weight=w_l,
                       steps_done=c * chunk, stop=stop, seconds=seconds,
                       best_val_loss=best["loss"])
        if discrete:
            R = rounded(P, grid_arr)
            summary.update(rounded_rho=metrics.rho(*R, n), frozen=frozen_fraction())
            np.savez(os.path.join(out, "rounded.npz"), H=R[0], K=R[1], F=R[2], n=n,
                     rho=summary["rounded_rho"])
        with open(os.path.join(out, "summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
        if run:
            run.summary.update({k: v for k, v in summary.items() if k not in cfg})
        return summary

    def frozen_fraction():
        return float(1.0 - sum(f.sum() for f in free) / sum(f.size for f in free))

    def freeze(c):
        """Snap to the grid, and freeze, the free entries within delta of it."""
        frac = (c * chunk / steps - freeze_start) / max(1e-12, 1.0 - freeze_start)
        delta = freeze_delta[0] + (freeze_delta[1] - freeze_delta[0]) * frac
        for p, f, q in zip(P, free, rounded(P, grid_arr)):
            hit = (f == 1.0) & (np.abs(p - q) < delta)
            p[hit] = q[hit]
            f[hit] = 0.0

    c = first - 1
    if stop is None:
        for c in range(first, n_chunks + 1):
            lr_now = lr_of(c)
            if discrete:
                adam_step = adam_steps_discrete(*P, *Mo, *V, adam_step, lr_now, clip, M, losses,
                                                quant * c / n_chunks, l1 * (1.0 - c / n_chunks),
                                                grid_arr, *free)
                if balance_terms:
                    balance(P)
                if freeze_every and (c * chunk) % freeze_every == 0 and \
                        c * chunk >= freeze_start * steps:
                    freeze(c)
            else:
                adam_step = adam_steps(*P, *Mo, *V, adam_step, lr_now, clip, M, losses)
            loss = exact_loss(*P, M)
            if schedule == "plateau":
                sched.step(loss)
            if loss < best["loss"]:
                if loss < best["loss"] * (1.0 - 1e-3):
                    last_gain = c
                best.update(P=[p.copy() for p in P], loss=loss, step=c * chunk)
            if discrete:
                # the pressures move the point away from the best one: run the whole budget,
                # unless the rounded point is already an exact decomposition
                if (c * chunk) % log_every == 0 and metrics.rho(*rounded(P, grid_arr), n) < 1e-12:
                    stop = "rounded"
                elif c == n_chunks:
                    stop = "budget"
            elif rho_of_loss(best["loss"], n) < stop_rho:
                stop = "rho"
            elif stall and (c - last_gain) * chunk >= stall and \
                    (schedule != "plateau" or sched.lr <= lr_min * (1 + 1e-12)):
                stop = "stall"
            elif c == n_chunks:
                stop = "budget"
            if (c * chunk) % log_every == 0 or stop:
                rho, w = rho_of_loss(loss, n), metrics.weight(*P)
                row = dict(lr=lr_now, loss=loss, rho=rho, weight=w,
                           best_rho=rho_of_loss(best["loss"], n))
                extra = ""
                if discrete:
                    row.update(frozen=frozen_fraction(),
                               rho_rounded=metrics.rho(*rounded(P, grid_arr), n))
                    extra = f",{row['frozen']!r},{row['rho_rounded']!r}"
                log.write(f"{c * chunk},{lr_now!r},{loss!r},{rho!r},{w!r}{extra}\n")
                if run:
                    run.log(row, step=c * chunk)
            if (c * chunk) % ckpt_every == 0 or stop:
                summary = save(c)
                if not quiet:
                    rate = (time.time() - t0) / ((c - first + 1) * chunk)
                    print(f"[{os.path.basename(out)}] step {c * chunk} lr {lr_now:.1e} rho "
                          f"{rho_of_loss(loss, n):.4e} best {summary['rho']:.4e} @ "
                          f"{best['step']} | {rate * 1e6:.1f} us/step"
                          + (f" | stop: {stop}" if stop else ""), flush=True)
            if stop or c == _interrupt_at:      # _interrupt_at: the tests emulate a killed run
                break
    log.close()
    summary_path = os.path.join(out, "summary.json")
    if polish and stop and os.path.exists(summary_path):
        with open(summary_path) as fh:
            summary = json.load(fh)
        if "polish_rho" not in summary:
            # with the discrete pressures, polish the last point (the best one by the loss
            # alone comes from before the pressures acted)
            start = P if discrete else best["P"]
            summary.update(_polish(out, dict(P=[p[..., None] for p in start]), cfg, polish))
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
    ap.add_argument("--n", type=int, default=4, help="matrix size")
    ap.add_argument("--rank", type=int, required=True, help="number of multiplications r")
    ap.add_argument("--seed", type=int, default=0, help="random start (without --init-from)")
    ap.add_argument("--init-from", metavar="FILE", help="start from the factors stored in FILE")
    ap.add_argument("--out", required=True, help="run directory")
    ap.add_argument("--steps", type=int, default=2_000_000, help="largest number of steps")
    ap.add_argument("--lr", type=float, default=1e-4, help="initial learning rate")
    ap.add_argument("--lr-min", type=float, default=1e-6, help="smallest learning rate")
    ap.add_argument("--lr-factor", type=float, default=0.5, help="plateau: reduction factor")
    ap.add_argument("--patience", type=int, default=10_000, help="plateau: patience (steps)")
    ap.add_argument("--schedule", choices=["plateau", "cosine", "constant"], default="plateau")
    ap.add_argument("--clip", type=float, default=CLIP, help="gradient-norm clipping")
    ap.add_argument("--chunk", type=int, default=100,
                    help="steps between evaluations of the schedule, the best point and the stops")
    ap.add_argument("--stop-rho", type=float, default=1e-8)
    ap.add_argument("--stall", type=int, default=100_000, help="0: never stop on a stall")
    ap.add_argument("--polish", type=int, default=0, metavar="N",
                    help="end with at most N Levenberg-Marquardt iterations")
    ap.add_argument("--extend", action="store_true",
                    help="continue from the checkpoint with other --steps, --stall, --lr-min, "
                         "also after a stop on the budget or a stall")
    ap.add_argument("--log-every", type=int, default=1000)
    ap.add_argument("--ckpt-every", type=int, default=200_000)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--wandb", metavar="ENTITY/PROJECT", help="log to Weights & Biases")
    ap.add_argument("--wandb-group")
    ap.add_argument("--wandb-tags", nargs="*")
    ap.add_argument("--wandb-job-type")
    g = ap.add_argument_group("pressures towards discrete solutions")
    g.add_argument("--quant", type=float, default=0.0, metavar="LAM",
                   help="penalty LAM * sum dist(theta, grid)^2, its weight growing from 0 to LAM")
    g.add_argument("--l1", type=float, default=0.0, metavar="LAM",
                   help="L1 proximal step, weight decreasing from LAM to 0")
    g.add_argument("--grid", type=float, nargs="+", default=list(GRID),
                   help="non-negative grid values (with signs)")
    g.add_argument("--balance", action="store_true",
                   help="rescale every term to ||h|| = ||k|| = ||f|| after every chunk")
    g.add_argument("--freeze-every", type=int, default=0, metavar="N",
                   help="every N steps, snap and freeze the entries within delta of the grid")
    g.add_argument("--freeze-delta", type=float, nargs=2, default=[0.05, 0.05],
                   metavar=("START", "END"), help="delta, linear from START to END")
    g.add_argument("--freeze-start", type=float, default=0.0,
                   help="fraction of the steps before freezing starts")
    a = ap.parse_args()
    best = train_exact(a.n, a.rank, a.out, a.seed, a.init_from, a.steps, a.lr, a.lr_min,
                       a.lr_factor, a.patience, a.schedule, a.clip, a.chunk, a.stop_rho, a.stall,
                       a.polish, a.log_every, a.ckpt_every, a.quiet, a.wandb, a.wandb_group,
                       a.wandb_tags, a.wandb_job_type, a.quant, a.l1, a.grid, a.balance,
                       a.freeze_every, a.freeze_delta, a.freeze_start, a.extend)
    print(f"best rho {rho_of_loss(best['loss'], a.n):.6e} at step {best['step']}")


if __name__ == "__main__":
    main()
