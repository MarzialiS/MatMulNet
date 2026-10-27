"""Block 2: 47 and 48 terms from random starts (N(0, 1) factors, no link to the known scheme).
Exact loss, plateau x0.5 / patience 10^4 steps down to 1e-8, initial lr 3e-4, 1e-3, 3e-3, 1e-2
(the configurations that solved at least one start at sigma = 0.6 by training alone, block 1),
at most 10^7 steps, stop at rho < 1e-8 or on a stall, then at most 200 LM iterations.
Seeds 1-10 for every lr and rank: 80 runs on the 8 P-cores. Run from the repository root."""
import json
import os
import sys

sys.path.insert(0, "experiments")
import phase1  # noqa: E402

D = "/home/marziali/MatMulNet/runs/round2-block2"
W = "sara-marziali-university-of-siena/matmulnet-4x4"
STEPS = 10_000_000
LRS = ["3e-4", "1e-3", "3e-3", "1e-2"]
RANKS = [47, 48]
SEEDS = range(1, 11)


class A:
    cpus = "p"
    jobs = None


jobs = []
for seed in SEEDS:
    for rank in RANKS:
        for lr in LRS:
            out = os.path.join(D, f"r{rank}", f"lr{lr}", f"seed{seed}")
            summary = os.path.join(out, "summary.json")
            if os.path.exists(summary) and "polish_rho" in json.load(open(summary)):
                continue
            os.makedirs(out, exist_ok=True)
            cmd = [sys.executable, "-m", "matmulnet.exact", "--n", "4", "--rank", str(rank),
                   "--seed", str(seed), "--out", out, "--steps", str(STEPS),
                   "--lr", lr, "--lr-min", "1e-8", "--patience", "10000",
                   "--polish", str(phase1.POLISH_ITER), "--log-every", "10000", "--quiet",
                   "--wandb", W, "--wandb-group", f"block2_r{rank}_lr{lr}",
                   "--wandb-job-type", "block2-scratch", "--wandb-tags", "round-2", "block-2",
                   f"r{rank}", f"lr{lr}"]
            jobs.append((f"r{rank}/lr{lr}/seed{seed}", cmd, os.path.join(out, "stdout.log")))

os.makedirs(D, exist_ok=True)
phase1.pinned_pool(A, jobs)
print("BLOCK 2 DONE", flush=True)
