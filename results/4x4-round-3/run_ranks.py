"""Round 3: runs from random starts (N(0, 1) factors, no link to any known scheme), at 46, 47, 48
and 49 terms. Exact loss, plateau x0.5 / patience 10^4 steps down to 1e-8, initial learning rate
3e-4, 1e-3, 3e-3, 1e-2 (the configurations that solved at least one start at sigma = 0.6 by
training alone, round 2, block 1), at most 10^7 steps, stop at rho < 1e-8 or on a stall, then at
most 200 LM iterations. Seeds 1-100 (or 1-N with --seeds N) for every rank and learning rate, on
the 8 P-cores.

    python results/4x4-round-3/run_ranks.py 46 49
    python results/4x4-round-3/run_ranks.py 60 64 --seeds 10

Run from the repository root; launched again, it skips what is done and resumes what was
interrupted."""
import json
import os
import sys

sys.path.insert(0, "experiments")
import phase1  # noqa: E402

D = "/home/marziali/MatMulNet/runs/round3-ranks"
W = "sara-marziali-university-of-siena/matmulnet-4x4"
STEPS = 10_000_000
LRS = ["3e-4", "1e-3", "3e-3", "1e-2"]
SEEDS = range(1, 101)


class A:
    cpus = "p"
    jobs = None


def rank_jobs(ranks, seeds=SEEDS):
    jobs = []
    for seed in seeds:
        for rank in ranks:
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
                       "--wandb", W, "--wandb-group", f"round3_r{rank}_lr{lr}",
                       "--wandb-job-type", "round3-ranks", "--wandb-tags", "round-3", "ranks",
                       f"r{rank}", f"lr{lr}"]
                jobs.append((f"r{rank}/lr{lr}/seed{seed}", cmd, os.path.join(out, "stdout.log")))
    return jobs


if __name__ == "__main__":
    args = sys.argv[1:]
    seeds = SEEDS
    if "--seeds" in args:
        i = args.index("--seeds")
        seeds = range(1, int(args[i + 1]) + 1)
        del args[i:i + 2]
    phase1.pinned_pool(A, rank_jobs([int(r) for r in args], seeds))
    print("RANKS DONE", flush=True)
