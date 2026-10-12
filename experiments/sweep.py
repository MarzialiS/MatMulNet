"""Run the training sweeps of the paper, several runs in parallel.

    python experiments/sweep.py calibration     # 2x2, r = 6, 7
    python experiments/sweep.py rank            # 3x3, r = 16, ..., 24
    python experiments/sweep.py border          # 3x3, r = 16, ..., 21, exponent ranges
                                                # [-2,4], [-3,5], [-4,6]
    python experiments/sweep.py rank4           # 4x4, r = 45, ..., 49
    python experiments/sweep.py border4         # 4x4, r = 44, ..., 47, exponent range [-1,2]

Each configuration is run 7 times (seeds 1-7); --ranks and --seeds select a subset or other
values. Every run is a single-threaded process writing
to runs/<name>/; finished runs are skipped and interrupted ones continue from their checkpoint,
so the command can simply be issued again.

Each run is pinned to one logical CPU, never sharing a physical core with another run (two runs on
the two hardware threads of a core are each almost twice as slow). --cpus chooses the CPUs:
"all" (default: one per physical core), "p" or "e" (only the performance or the efficiency cores
of a hybrid Intel CPU), or a list such as 0,2,4-7. On a hybrid CPU the performance cores are
filled first. --jobs (default: as many as the CPUs) runs fewer at once, e.g. on a shared machine.
"""
import argparse
import json
import os
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = range(1, 8)
# experiment: (n, ranks, exponent ranges of the border-rank model or None)
EXPERIMENTS = {
    "calibration": (2, (6, 7), None),
    "rank": (3, range(16, 25), None),
    "border": (3, range(16, 22), [(-2, 4), (-3, 5), (-4, 6)]),
    "rank4": (4, range(45, 50), None),
    "border4": (4, range(44, 48), [(-1, 2)]),
}


def configurations(experiment, init="normal", ranks=None, seeds=None):
    n, default_ranks, ranges = EXPERIMENTS[experiment]
    ranks, seeds = ranks or default_ranks, seeds or SEEDS
    if ranges is None:
        return [(f"rank_n{n}_r{r}_seed{s}", ["--n", str(n), "--rank", str(r), "--seed", str(s)])
                for r in ranks for s in seeds]
    tag = "" if init == "normal" else f"_{init}"
    return [(f"border_n{n}_r{r}_d{a}_{b}{tag}_seed{s}",
             ["--n", str(n), "--rank", str(r), "--degrees", str(a), str(b), "--seed", str(s),
              "--init", init])
            for r in ranks for a, b in ranges for s in seeds]


def _cpu_list(text):
    """"0,2,4-7" -> [0, 2, 4, 5, 6, 7]"""
    out = []
    for part in text.strip().split(","):
        if part:
            a, _, b = part.partition("-")
            out += range(int(a), int(b or a) + 1)
    return out


def _read_cpus(path):
    try:
        with open(path) as fh:
            return _cpu_list(fh.read())
    except OSError:
        return []


def cpu_pool(spec):
    """Logical CPUs to pin the runs to, at most one per physical core, performance cores first."""
    allowed = sorted(os.sched_getaffinity(0))
    if spec not in ("all", "p", "e"):
        return [c for c in _cpu_list(spec) if c in allowed]
    first_thread = [c for c in allowed if _read_cpus(
        f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list")[:1] in ([c], [])]
    perf = set(_read_cpus("/sys/devices/cpu_core/cpus"))       # empty unless hybrid Intel
    eff = set(_read_cpus("/sys/devices/cpu_atom/cpus"))
    if spec == "p":
        return [c for c in first_thread if c in perf] or first_thread
    if spec == "e":
        return [c for c in first_thread if c in eff]
    return [c for c in first_thread if c in perf] + [c for c in first_thread if c not in perf]


def finished(run_dir, max_epochs):
    try:
        with open(os.path.join(run_dir, "summary.json")) as fh:
            return json.load(fh)["epochs_done"] >= max_epochs
    except (OSError, KeyError, ValueError):
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=list(EXPERIMENTS))
    ap.add_argument("--ranks", type=int, nargs="+", help="ranks to run (default: all)")
    ap.add_argument("--seeds", type=int, nargs="+", help="seeds to run (default: 1 to 7)")
    ap.add_argument("--wandb", metavar="ENTITY/PROJECT", help="log the runs to Weights & Biases")
    ap.add_argument("--cpus", default="all",
                    help='CPUs for the runs: "all" (one per physical core), "p", "e", or a list '
                         'such as 0,2,4-7 (default: all)')
    ap.add_argument("--jobs", type=int, help="runs in parallel (default: one per CPU of --cpus)")
    ap.add_argument("--runs-dir", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--max-epochs", type=int, default=1_000_000)
    ap.add_argument("--backend", choices=["numba", "torch"], default="numba")
    ap.add_argument("--init", choices=["normal", "scaled"], default="normal",
                    help="initialisation of the border-rank model (see matmulnet.train)")
    a = ap.parse_args()
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               NUMBA_NUM_THREADS="1")
    os.makedirs(a.runs_dir, exist_ok=True)
    cpus = cpu_pool(a.cpus)
    if not cpus:
        ap.error(f"no usable CPU in --cpus {a.cpus}")
    jobs = min(a.jobs or len(cpus), len(cpus))
    free = queue.Queue()
    for c in cpus[:jobs]:
        free.put(c)
    print(f"{jobs} runs at a time, on CPUs {','.join(map(str, cpus[:jobs]))}", flush=True)

    def run(cfg):
        name, args = cfg
        run_dir = os.path.join(a.runs_dir, name)
        if finished(run_dir, a.max_epochs):
            return name, "already done"
        os.makedirs(run_dir, exist_ok=True)
        cmd = [sys.executable, "-m", "matmulnet.train", *args, "--out", run_dir, "--max-epochs",
               str(a.max_epochs), "--backend", a.backend, "--quiet"]
        if a.wandb:
            cmd += ["--wandb", a.wandb]
        cpu = free.get()
        try:
            with open(os.path.join(run_dir, "stdout.log"), "a") as log:
                code = subprocess.call(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       preexec_fn=lambda: os.sched_setaffinity(0, {cpu}))
        finally:
            free.put(cpu)
        return name, "done" if code == 0 else f"FAILED (exit code {code})"

    todo = configurations(a.experiment, a.init, a.ranks, a.seeds)
    with ThreadPoolExecutor(jobs) as pool:
        for i, (name, status) in enumerate(pool.map(run, todo), 1):
            print(f"[{i}/{len(todo)}] {name}: {status}", flush=True)


if __name__ == "__main__":
    main()
