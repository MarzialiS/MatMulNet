"""Direct constrained refinement of the six long runs (as in round 1): bounds starting at the
smallest integer that does not clip the final point, then 10 and 25, at most one hour per level,
one single-threaded process per P-core. Run from the repository root."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, "experiments")
import phase1  # noqa: E402

D = "/home/marziali/MatMulNet/runs/round2-block2-long"
RUNS = ["r48/lr1e-2/seed3", "r48/lr1e-3/seed3", "r48/lr3e-4/seed7", "r48/lr3e-3/seed5",
        "r47/lr3e-4/seed2", "r47/lr3e-4/seed7"]


class A:
    cpus = "p"
    jobs = len(RUNS)


jobs = []
for name in RUNS:
    init = os.path.join(D, name, "best.npz")
    z = np.load(init)
    first = math.ceil(max(np.abs(z[k]).max() for k in "HKF"))
    levels = [first] + [c for c in (10, 25) if c > first]
    out = os.path.join(D, "refine", name)
    if os.path.exists(os.path.join(out, "frontier.csv")):
        continue
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cmd = [sys.executable, "-m", "refinement.refine", "--init", init, "--out", out,
           "--levels", *map(str, levels), "--max-seconds", "3600", "--threads", "1"]
    jobs.append((name, cmd, os.path.join(D, "refine", name.replace("/", "_") + ".log")))

phase1.pinned_pool(A, jobs)
print("REFINE DONE", flush=True)
