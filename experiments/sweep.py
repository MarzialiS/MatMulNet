"""Run the training sweeps of the paper, several runs in parallel.

    python experiments/sweep.py calibration     # 2x2, r = 6, 7
    python experiments/sweep.py rank            # 3x3, r = 16, ..., 24
    python experiments/sweep.py border          # 3x3, r = 16, ..., 21, exponent ranges
                                                # [-2,4], [-3,5], [-4,6]

Each configuration is run 7 times (seeds 1-7). Every run is a single-threaded process writing
to runs/<name>/; finished runs are skipped and interrupted ones continue from their checkpoint,
so the command can simply be issued again.
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = range(1, 8)
RANGES = [(-2, 4), (-3, 5), (-4, 6)]


def configurations(experiment):
    if experiment == "calibration":
        return [(f"rank_n2_r{r}_seed{s}", ["--n", "2", "--rank", str(r), "--seed", str(s)])
                for r in (6, 7) for s in SEEDS]
    if experiment == "rank":
        return [(f"rank_n3_r{r}_seed{s}", ["--n", "3", "--rank", str(r), "--seed", str(s)])
                for r in range(16, 25) for s in SEEDS]
    return [(f"border_n3_r{r}_d{a}_{b}_seed{s}",
             ["--n", "3", "--rank", str(r), "--degrees", str(a), str(b), "--seed", str(s)])
            for r in range(16, 22) for a, b in RANGES for s in SEEDS]


def finished(run_dir, max_epochs):
    try:
        with open(os.path.join(run_dir, "summary.json")) as fh:
            return json.load(fh)["epochs_done"] >= max_epochs
    except (OSError, KeyError, ValueError):
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=["calibration", "rank", "border"])
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                    help="runs in parallel (default: half the logical cores)")
    ap.add_argument("--runs-dir", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--max-epochs", type=int, default=1_000_000)
    ap.add_argument("--backend", choices=["numba", "torch"], default="numba")
    a = ap.parse_args()
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               NUMBA_NUM_THREADS="1")
    os.makedirs(a.runs_dir, exist_ok=True)

    def run(cfg):
        name, args = cfg
        run_dir = os.path.join(a.runs_dir, name)
        if finished(run_dir, a.max_epochs):
            return name, "already done"
        os.makedirs(run_dir, exist_ok=True)
        cmd = [sys.executable, "-m", "matmulnet.train", *args, "--out", run_dir, "--max-epochs",
               str(a.max_epochs), "--backend", a.backend, "--quiet"]
        with open(os.path.join(run_dir, "stdout.log"), "a") as log:
            code = subprocess.call(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        return name, "done" if code == 0 else f"FAILED (exit code {code})"

    todo = configurations(a.experiment)
    with ThreadPoolExecutor(a.jobs) as pool:
        for i, (name, status) in enumerate(pool.map(run, todo), 1):
            print(f"[{i}/{len(todo)}] {name}: {status}", flush=True)


if __name__ == "__main__":
    main()
