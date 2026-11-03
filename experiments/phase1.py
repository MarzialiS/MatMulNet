"""Round 2, phase 1: recover a known 48-term decomposition of M_4 from perturbed starting points.

    python experiments/phase1.py starts --scheme schemes/rational48.npz
    python experiments/phase1.py lm                                      # LM-only control
    python experiments/phase1.py train --tag protocol --max-epochs 50000  # training + polish
    python experiments/phase1.py train --tag cos1e-3 --max-epochs 50000 -- --schedule cosine --lr 1e-3
    python experiments/phase1.py table

Starting points (`starts`), for seeds 1-5:
  noise   the scheme plus Gaussian noise of standard deviation sigma times the r.m.s. of its
          non-zero coefficients, sigma in 0.1, 0.2, 0.3, 0.5, 0.7, 1;
  interp  (1 - t) times the scheme plus t times a random point drawn as a training run draws its
          initial point (N(0, 1)), t in 0.1, ..., 0.9.
A start is "solved" when the end point has rho < 1e-10 with weight W below W_MAX (a bounded
decomposition, not a border-rank degeneration with rho falling while W grows).

`lm` is the control: unconstrained Levenberg-Marquardt from the starting point alone, with a
budget well above the one of the polish that ends a training run. A training configuration
counts only on the starts where this control fails. `train` runs matmulnet.train from every
start (arguments after `--` are passed on), ending with --polish. Every job is a single-threaded
process pinned to its own core (--cpus, as in experiments/sweep.py); finished jobs are skipped.
`reduce` (rank reduction) starts from the last points of a training configuration with more
terms (--source) and, one term at a time, drops the term whose removal increases rho the least
and trains --stage-epochs epochs, down to --target terms, then polishes.
`table` prints, per start level, the fraction of solved starts for the control and for every
training configuration, and whether the solution found is the scheme itself (same rank-one
terms up to order) or another decomposition.
"""
import argparse
import glob
import json
import os
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from known_schemes import interpolate, perturb  # noqa: E402
from sweep import cpu_pool  # noqa: E402

from matmulnet import metrics  # noqa: E402

SIGMAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5]
TS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SEEDS = [1, 2, 3, 4, 5]
RHO_OK, W_MAX = 1e-10, 1e4         # W of the rational 48-term scheme: 210.5
LM_ITER, LM_SECONDS, POLISH_ITER = 1000, 900, 200
DEFAULT_DIR = os.path.join(ROOT, "runs", "round2-phase1")
ENV = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
           NUMBA_NUM_THREADS="1")


def start_names(kinds=("noise", "interp"), seeds=SEEDS):
    names = []
    if "noise" in kinds:
        names += [f"noise_s{s:g}_seed{i}" for s in SIGMAS for i in seeds]
    if "interp" in kinds:
        names += [f"interp_t{t:g}_seed{i}" for t in TS for i in seeds]
    return names


def level(name):
    return name.rsplit("_seed", 1)[0]


def same_terms(H, K, F, ref):
    """Largest relative distance between the rank-one terms of (H, K, F) and those of the
    reference, after the best matching of the terms: invariant under the rescalings of each
    term and under their order. Small means the reference scheme itself has been found."""
    from scipy.optimize import linear_sum_assignment
    t = np.einsum("as,bs,sc->sabc", H, K, F).reshape(H.shape[1], -1)
    t0 = np.einsum("as,bs,sc->sabc", *ref).reshape(ref[0].shape[1], -1)
    d = np.linalg.norm(t[:, None, :] - t0[None, :, :], axis=2)
    i, j = linear_sum_assignment(d)
    return float(np.max(d[i, j] / np.linalg.norm(t0[j], axis=1)))


def solved(rho, w):
    return rho < RHO_OK and w < W_MAX


def cmd_starts(a):
    z = np.load(a.scheme)
    coeff = [z[k] for k in ("H_coeff", "K_coeff", "F_coeff")]
    out = os.path.join(a.dir, "starts")
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "scheme.npz"), H=coeff[0][..., 0], K=coeff[1][..., 0],
             F=coeff[2][..., 0], n=4)
    for name in selected(a):
        if os.path.exists(os.path.join(out, name + ".npz")):
            continue
        kind, rest = name.split("_", 1)
        value, seed = float(rest.split("_")[0][1:]), int(name.rsplit("seed", 1)[1])
        rng = np.random.default_rng(seed)
        c = perturb(coeff, value, rng) if kind == "noise" else interpolate(coeff, value, rng)
        H, K, F = (x[..., 0] for x in c)
        np.savez(os.path.join(out, name + ".npz"), H_coeff=c[0], K_coeff=c[1], F_coeff=c[2],
                 degrees=np.array([0]), H=H, K=K, F=F, n=4)
        print(f"{name}: rho {metrics.rho(H, K, F, 4):.3f}, W {metrics.weight(H, K, F):.1f}")


def pinned_pool(a, jobs):
    """Run the command lines in `jobs` [(label, cmd, log)], each pinned to one free CPU."""
    cpus = cpu_pool(a.cpus)
    n = min(a.jobs or len(cpus), len(cpus))
    free = queue.Queue()
    for c in cpus[:n]:
        free.put(c)
    print(f"{len(jobs)} jobs, {n} at a time on CPUs {','.join(map(str, cpus[:n]))}", flush=True)

    def run(job):
        label, cmd, log = job
        cpu = free.get()
        try:
            with open(log, "a") as fh:
                code = subprocess.call(cmd, cwd=ROOT, env=ENV, stdout=fh, stderr=subprocess.STDOUT,
                                       preexec_fn=lambda: os.sched_setaffinity(0, {cpu}))
        finally:
            free.put(cpu)
        return label, code

    with ThreadPoolExecutor(n) as pool:
        for i, (label, code) in enumerate(pool.map(run, jobs), 1):
            print(f"[{i}/{len(jobs)}] {label}: {'done' if code == 0 else f'FAILED ({code})'}",
                  flush=True)


def selected(a):
    names = start_names(a.kinds, a.seeds)
    if a.levels:
        names = [n for n in names if level(n) in a.levels]
    return names


def cmd_lm(a):
    jobs = []
    for name in selected(a):
        out = os.path.join(a.dir, "lm", name)
        if os.path.exists(os.path.join(out, "result.json")):
            continue
        os.makedirs(out, exist_ok=True)
        cmd = [sys.executable, os.path.abspath(__file__), "lm-one", name, "--dir", a.dir]
        if a.wandb:
            cmd += ["--wandb", a.wandb]
            if a.wandb_tags:
                cmd += ["--wandb-tags", *a.wandb_tags]
        jobs.append((name, cmd, os.path.join(out, "stdout.log")))
    pinned_pool(a, jobs)


def cmd_lm_one(a):
    from refinement.refine import Problem, projected_lm
    z = np.load(os.path.join(a.dir, "starts", a.name + ".npz"))
    prob = Problem(4, z["H"].shape[1])
    t0 = time.time()
    x, status, iters = projected_lm(prob, prob.pack(z["H"], z["K"], z["F"]), np.inf, LM_ITER,
                                    max_seconds=LM_SECONDS)
    H, K, F = prob.unpack(x)
    ref = np.load(os.path.join(a.dir, "starts", "scheme.npz"))
    res = dict(start=a.name, level=level(a.name), rho=metrics.rho(H, K, F, 4),
               weight=metrics.weight(H, K, F), status=status, iterations=iters,
               seconds=time.time() - t0, terms_vs_scheme=same_terms(H, K, F, [ref[k] for k in "HKF"]))
    res["solved"] = solved(res["rho"], res["weight"])
    out = os.path.join(a.dir, "lm", a.name)
    np.savez(os.path.join(out, "end.npz"), H=H, K=K, F=F, n=4)
    with open(os.path.join(out, "result.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res))
    if a.wandb:
        import wandb
        entity, _, project = a.wandb.rpartition("/")
        run = wandb.init(entity=entity or None, project=project, name=f"lm_{a.name}",
                         group=f"phase1_lm_{level(a.name)}", job_type="phase1-lm-only",
                         tags=a.wandb_tags or ["round-2", "phase-1"],
                         config=dict(start=a.name, level=level(a.name), max_iter=LM_ITER),
                         settings=wandb.Settings(console="off"))
        run.summary.update(res)
        run.finish()


def cmd_train(a):
    jobs = []
    for name in selected(a):
        out = os.path.join(a.dir, "train", a.tag, name)
        summary = os.path.join(out, "summary.json")
        if os.path.exists(summary) and "polish_rho" in json.load(open(summary)):
            continue
        os.makedirs(out, exist_ok=True)
        seed = int(name.rsplit("seed", 1)[1])
        cmd = [sys.executable, "-m", "matmulnet.train", "--n", "4", "--rank", "48", "--seed",
               str(seed), "--init-from", os.path.join(a.dir, "starts", name + ".npz"), "--out", out,
               "--max-epochs", str(a.max_epochs), "--polish", str(POLISH_ITER), "--quiet",
               "--ckpt-every", "5000", *a.extra]
        if a.wandb:
            cmd += ["--wandb", a.wandb, "--wandb-group", f"phase1_{a.tag}_{level(name)}",
                    "--wandb-job-type", "phase1-train", "--wandb-tags", "round-2", "phase-1", a.tag]
        jobs.append((f"{a.tag}/{name}", cmd, os.path.join(out, "stdout.log")))
    pinned_pool(a, jobs)


def cmd_exact(a):
    jobs = []
    for name in selected(a):
        out = os.path.join(a.dir, "exact", a.tag, name)
        summary = os.path.join(out, "summary.json")
        if os.path.exists(summary) and "polish_rho" in json.load(open(summary)):
            continue
        os.makedirs(out, exist_ok=True)
        cmd = [sys.executable, "-m", "matmulnet.exact", "--n", "4", "--rank", "48",
               "--init-from", os.path.join(a.dir, "starts", name + ".npz"), "--out", out,
               "--polish", str(POLISH_ITER), "--quiet", *a.extra]
        if a.wandb:
            cmd += ["--wandb", a.wandb, "--wandb-group", f"phase1b_{a.tag}_{level(name)}",
                    "--wandb-job-type", "phase1b-exact", "--wandb-tags", "round-2", "phase-1b",
                    a.tag]
        jobs.append((f"{a.tag}/{name}", cmd, os.path.join(out, "stdout.log")))
    pinned_pool(a, jobs)


def drop_least_damage(H, K, F):
    """Remove the term whose removal increases the residual rho the least."""
    R = metrics.tensor(H, K, F) - metrics.matmul_tensor(4)
    inner = np.einsum("abc,as,bs,sc->s", R, H, K, F, optimize=True)
    sq = (H ** 2).sum(0) * (K ** 2).sum(0) * (F ** 2).sum(1)
    s = int(np.argmin(sq - 2 * inner))         # ||R - t_s||^2 - ||R||^2
    keep = np.arange(H.shape[1]) != s
    return H[:, keep], K[:, keep], F[keep]


def cmd_reduce(a):
    jobs = []
    for name in selected(a):
        if not os.path.exists(os.path.join(a.dir, "train", a.source, name, "last.npz")):
            continue
        final = os.path.join(a.dir, "reduce", a.tag, name, f"r{a.target}", "summary.json")
        if os.path.exists(final) and "polish_rho" in json.load(open(final)):
            continue
        os.makedirs(os.path.dirname(final), exist_ok=True)
        cmd = [sys.executable, os.path.abspath(__file__), "reduce-one", name, "--dir", a.dir,
               "--tag", a.tag, "--source", a.source, "--target", str(a.target),
               "--stage-epochs", str(a.stage_epochs), "--stage-lr", str(a.stage_lr)]
        if a.wandb:
            cmd += ["--wandb", a.wandb]
        jobs.append((f"reduce {a.tag}/{name}", cmd,
                     os.path.join(os.path.dirname(os.path.dirname(final)), "stdout.log")))
    pinned_pool(a, jobs)


def cmd_reduce_one(a):
    """Rank reduction from the last point of a run with more terms: drop the term whose removal
    hurts least, train for --stage-epochs (cosine from --stage-lr), repeat down to --target,
    then polish."""
    from matmulnet.train import train
    z = np.load(os.path.join(a.dir, "train", a.source, a.name, "last.npz"))
    H, K, F = z["H"], z["K"], z["F"]
    seed = int(a.name.rsplit("seed", 1)[1])
    base = os.path.join(a.dir, "reduce", a.tag, a.name)
    while H.shape[1] > a.target:
        H, K, F = drop_least_damage(H, K, F)
        r = H.shape[1]
        stage = os.path.join(base, f"r{r}")
        os.makedirs(stage, exist_ok=True)
        start = os.path.join(stage, "start.npz")
        np.savez(start, H_coeff=H[..., None], K_coeff=K[..., None], F_coeff=F[..., None],
                 degrees=np.array([0]), H=H, K=K, F=F, n=4, rho=metrics.rho(H, K, F, 4))
        train(4, r, stage, seed=seed, max_epochs=a.stage_epochs, init_from=start, quiet=True,
              ckpt_every=5000, schedule="cosine", lr=a.stage_lr,
              polish=POLISH_ITER if r == a.target else 0, wandb=a.wandb,
              wandb_group=f"phase1_reduce_{a.tag}_{level(a.name)}", wandb_job_type="phase1-reduce",
              wandb_tags=["round-2", "phase-1", f"reduce_{a.tag}"])
        rho_start = float(np.load(start)["rho"])
        z = np.load(os.path.join(stage, "last.npz"))
        H, K, F = z["H"], z["K"], z["F"]
        print(f"{a.name} r={r}: rho {rho_start:.3e} after the removal, "
              f"{metrics.rho(H, K, F, 4):.3e} at the end of the stage", flush=True)


def collect(a):
    """{(method, level): [result per start]}; method 'lm' or the training tag."""
    ref = np.load(os.path.join(a.dir, "starts", "scheme.npz"))
    ref = [ref[k] for k in "HKF"]
    rows = {}
    for f in glob.glob(os.path.join(a.dir, "lm", "*", "result.json")):
        r = json.load(open(f))
        rows.setdefault(("lm", r["level"]), []).append(r)
    for f in glob.glob(os.path.join(a.dir, "train", "*", "*", "summary.json")) + \
            glob.glob(os.path.join(a.dir, "exact", "*", "*", "summary.json")) + \
            glob.glob(os.path.join(a.dir, "reduce", "*", "*", "r*", "summary.json")):
        s = json.load(open(f))
        if "polish_rho" not in s:
            continue
        run_dir = os.path.dirname(f)
        if os.sep + "reduce" + os.sep in run_dir:
            tag, name = run_dir.split(os.sep)[-3:-1]
            tag = "reduce:" + tag
        else:
            tag, name = run_dir.split(os.sep)[-2:]
            if os.sep + "exact" + os.sep in run_dir:
                tag = "x:" + tag
        cache = os.path.join(run_dir, "terms_vs_scheme.json")
        if not os.path.exists(cache):
            z = np.load(os.path.join(run_dir, "polished.npz"))
            json.dump(same_terms(z["H"], z["K"], z["F"], ref), open(cache, "w"))
        r = dict(start=name, level=level(name), rho=s["polish_rho"], weight=s["polish_weight"],
                 train_rho=s["last_rho"] if s.get("discrete") else s["rho"],
                 rounded_rho=s.get("rounded_rho"), best_val_loss=s["best_val_loss"],
                 seconds=s["seconds"],
                 terms_vs_scheme=json.load(open(cache)), steps=s.get("steps_done"),
                 stop=s.get("stop"))
        r["solved"] = solved(r["rho"], r["weight"])
        rows.setdefault((tag, r["level"]), []).append(r)
    return rows


def cmd_table(a):
    """One row per method, one column per start level: solved / runs, with the number of
    starts that training alone brings below 1e-8 (t) and the median rho at the end."""
    import fnmatch
    rows = collect(a)
    if a.seeds != SEEDS or a.levels:
        keep = set(selected(a))
        rows = {k: [r for r in rs if r["start"] in keep] for k, rs in rows.items()}
        rows = {k: rs for k, rs in rows.items() if rs}
    methods = ["lm"] + sorted({m for m, _ in rows} - {"lm"})
    if a.methods:
        methods = [m for m in methods if any(fnmatch.fnmatch(m, p) for p in a.methods)]
    levels = [lv for lv in dict.fromkeys(level(n) for n in start_names()) if
              any((m, lv) in rows for m in methods)]
    width = max(len(m) for m in methods) + 2
    print("solved / runs [t: training alone reaches rho < 1e-8], median rho at the end")
    print(f"{'method':<{width}}" + "".join(f"{lv:>24}" for lv in levels))
    for m in methods:
        cells = []
        for lv in levels:
            rs = rows.get((m, lv), [])
            if not rs:
                cells.append(f"{'-':>24}")
                continue
            ok = sum(r["solved"] for r in rs)
            alone = "" if m == "lm" else f" [{sum(r['train_rho'] < 1e-8 for r in rs)}t]"
            med = np.median([r["rho"] for r in rs])
            cells.append(f"{ok}/{len(rs)}{alone} {med:8.1e}".rjust(24))
        print(f"{m:<{width}}" + "".join(cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["starts", "lm", "lm-one", "train", "exact", "reduce",
                                        "reduce-one", "table"])
    ap.add_argument("name", nargs="?", help="lm-one, reduce-one: the starting point")
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--scheme", default=os.path.join(ROOT, "schemes", "rational48.npz"))
    ap.add_argument("--kinds", nargs="+", default=["noise", "interp"])
    ap.add_argument("--levels", nargs="+", help="e.g. noise_s0.5 interp_t0.3 (default: all)")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--tag", default="protocol", help="train: name of the configuration")
    ap.add_argument("--max-epochs", type=int, default=50_000)
    ap.add_argument("--cpus", default="p")
    ap.add_argument("--jobs", type=int)
    ap.add_argument("--wandb", metavar="ENTITY/PROJECT")
    ap.add_argument("--wandb-tags", nargs="+", help="lm, lm-one: W&B tags (default round-2 phase-1)")
    ap.add_argument("--source", help="reduce: training tag whose last points have more terms")
    ap.add_argument("--target", type=int, default=48, help="reduce: final number of terms")
    ap.add_argument("--stage-epochs", type=int, default=10_000, help="reduce: epochs per term")
    ap.add_argument("--stage-lr", type=float, default=1e-3, help="reduce: initial learning rate")
    ap.add_argument("--methods", nargs="+", help="table: methods to show (glob patterns)")
    argv = sys.argv[1:]
    cut = argv.index("--") if "--" in argv else len(argv)
    a = ap.parse_args(argv[:cut])
    a.extra = argv[cut + 1:]          # train, exact: options passed on to the trainer
    {"starts": cmd_starts, "lm": cmd_lm, "lm-one": cmd_lm_one, "train": cmd_train,
     "exact": cmd_exact,
     "reduce": cmd_reduce, "reduce-one": cmd_reduce_one, "table": cmd_table}[a.command](a)


if __name__ == "__main__":
    main()
