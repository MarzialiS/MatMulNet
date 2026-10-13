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

SIGMAS = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
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
    for name in start_names():
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
                         tags=["round-2", "phase-1"], config=dict(start=a.name, level=level(a.name),
                                                                    max_iter=LM_ITER),
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


def collect(a):
    """{(method, level): [result per start]}; method 'lm' or the training tag."""
    ref = np.load(os.path.join(a.dir, "starts", "scheme.npz"))
    ref = [ref[k] for k in "HKF"]
    rows = {}
    for f in glob.glob(os.path.join(a.dir, "lm", "*", "result.json")):
        r = json.load(open(f))
        rows.setdefault(("lm", r["level"]), []).append(r)
    for f in glob.glob(os.path.join(a.dir, "train", "*", "*", "summary.json")):
        s = json.load(open(f))
        if "polish_rho" not in s:
            continue
        run_dir = os.path.dirname(f)
        tag, name = run_dir.split(os.sep)[-2:]
        cache = os.path.join(run_dir, "terms_vs_scheme.json")
        if not os.path.exists(cache):
            z = np.load(os.path.join(run_dir, "polished.npz"))
            json.dump(same_terms(z["H"], z["K"], z["F"], ref), open(cache, "w"))
        r = dict(start=name, level=level(name), rho=s["polish_rho"], weight=s["polish_weight"],
                 train_rho=s["rho"], best_val_loss=s["best_val_loss"], seconds=s["seconds"],
                 terms_vs_scheme=json.load(open(cache)))
        r["solved"] = solved(r["rho"], r["weight"])
        rows.setdefault((tag, r["level"]), []).append(r)
    return rows


def cmd_table(a):
    rows = collect(a)
    methods = ["lm"] + sorted({m for m, _ in rows} - {"lm"})
    levels = [lv for lv in dict.fromkeys(level(n) for n in start_names()) if
              any((m, lv) in rows for m in methods)]
    print("solved / runs  (s = the scheme itself, o = another decomposition); median rho at the end")
    print(f"{'level':<12}" + "".join(f"{m:>26}" for m in methods))
    for lv in levels:
        cells = []
        for m in methods:
            rs = rows.get((m, lv), [])
            if not rs:
                cells.append(f"{'-':>26}")
                continue
            ok = [r for r in rs if r["solved"]]
            same = sum(r["terms_vs_scheme"] < 1e-6 for r in ok)
            med = np.median([r["rho"] for r in rs])
            cells.append(f"{len(ok)}/{len(rs)} ({same}s {len(ok) - same}o) {med:9.1e}".rjust(26))
        print(f"{lv:<12}" + "".join(cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["starts", "lm", "lm-one", "train", "table"])
    ap.add_argument("name", nargs="?", help="lm-one: the starting point")
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
    argv = sys.argv[1:]
    cut = argv.index("--") if "--" in argv else len(argv)
    a = ap.parse_args(argv[:cut])
    a.extra = argv[cut + 1:]          # train: options passed on to matmulnet.train
    {"starts": cmd_starts, "lm": cmd_lm, "lm-one": cmd_lm_one, "train": cmd_train,
     "table": cmd_table}[a.command](a)


if __name__ == "__main__":
    main()
