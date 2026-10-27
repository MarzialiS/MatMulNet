"""Block 2, long continuation: six runs that ended lowest and were still going down at 10^7 steps,
continued from their checkpoints (copies in runs/round2-block2-long/, the originals are kept) up to
10^10 steps, with no stop on a stall and the learning rate held at the value it had reached.
Run from the repository root; launched again, the runs continue from their checkpoints."""
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, "experiments")
import phase1  # noqa: E402

SRC = "/home/marziali/MatMulNet/runs/round2-block2"
D = "/home/marziali/MatMulNet/runs/round2-block2-long"
W = "sara-marziali-university-of-siena/matmulnet-4x4"
STEPS = 10_000_000_000
RUNS = [(48, "1e-2", 3), (48, "1e-3", 3), (48, "3e-4", 7), (48, "3e-3", 5), (47, "3e-4", 2),
        (47, "3e-4", 7)]


class A:
    cpus = "p"
    jobs = len(RUNS)


jobs = []
for rank, lr, seed in RUNS:
    name = f"r{rank}/lr{lr}/seed{seed}"
    out = os.path.join(D, name)
    if not os.path.exists(out):
        shutil.copytree(os.path.join(SRC, name), out,
                        ignore=shutil.ignore_patterns("wandb_id.txt", "polished.npz"))
    lr_now = json.loads(str(np.load(os.path.join(SRC, name, "ckpt.npz"))["meta"]))["lr"]
    cmd = [sys.executable, "-m", "matmulnet.exact", "--n", "4", "--rank", str(rank),
           "--seed", str(seed), "--out", out, "--steps", str(STEPS), "--extend", "--stall", "0",
           "--lr", lr, "--lr-min", repr(lr_now), "--patience", "10000",
           "--polish", str(phase1.POLISH_ITER), "--log-every", "10000", "--quiet",
           "--wandb", W, "--wandb-group", f"block2long_r{rank}_lr{lr}",
           "--wandb-job-type", "block2-long", "--wandb-tags", "round-2", "block-2-long",
           f"r{rank}", f"lr{lr}"]
    jobs.append((name, cmd, os.path.join(out, "stdout.log")))

os.makedirs(D, exist_ok=True)
phase1.pinned_pool(A, jobs)
print("BLOCK 2 LONG DONE", flush=True)
