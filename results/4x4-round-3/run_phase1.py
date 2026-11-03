"""Round 3, phase 1: recovering the known 48-term scheme from perturbed copies.

Starts: the rational scheme of Dumas-Pernet-Sedoglavic plus Gaussian noise, sigma = 0.4, 0.5,
0.6, 0.7, 0.8, 1.0 (times the r.m.s. of its non-zero coefficients), seeds 1-100: 600 starts.
For every start:
  1. training on the exact loss, two configurations (plateau x0.5 / patience 10^4 steps down to
     1e-8, initial learning rate 3e-3 and 1e-2), at most 10^6 steps, stop at rho < 1e-8 or on a
     stall, then at most 200 Levenberg-Marquardt iterations;
  2. Levenberg-Marquardt alone, 1000 iterations (the control).
Solved: rho < 1e-10 and W < 1e4 at the end. Run from the repository root; launched again, it
skips what is done and resumes what was interrupted."""
import json
import os
import subprocess
import sys

sys.path.insert(0, "experiments")
import phase1  # noqa: E402

D = "/home/marziali/MatMulNet/runs/round3-phase1"
W = "sara-marziali-university-of-siena/matmulnet-4x4"
SCHEME = "results/4x4-round-1/known_schemes/rational48.npz"
LEVELS = ["noise_s0.4", "noise_s0.5", "noise_s0.6", "noise_s0.7", "noise_s0.8", "noise_s1"]
SEEDS = [str(s) for s in range(1, 101)]
CONFIGS = {"lr3e-3": ["--lr", "3e-3"], "lr1e-2": ["--lr", "1e-2"]}
COMMON = ["--lr-min", "1e-8", "--patience", "10000", "--steps", "1000000"]


class A:
    cpus = "p"
    jobs = None


def run_phase1(*args):
    subprocess.check_call([sys.executable, "experiments/phase1.py", *args, "--dir", D,
                           "--levels", *LEVELS, "--seeds", *SEEDS])


# 1. the starts
run_phase1("starts", "--scheme", SCHEME, "--kinds", "noise")

# 2. training, both configurations in one pool, seed by seed so that every level advances
jobs = []
for seed in SEEDS:
    for lv in LEVELS:
        for tag, extra in CONFIGS.items():
            name = f"{lv}_seed{seed}"
            out = os.path.join(D, "exact", tag, name)
            summary = os.path.join(out, "summary.json")
            if os.path.exists(summary) and "polish_rho" in json.load(open(summary)):
                continue
            os.makedirs(out, exist_ok=True)
            cmd = [sys.executable, "-m", "matmulnet.exact", "--n", "4", "--rank", "48",
                   "--init-from", os.path.join(D, "starts", name + ".npz"), "--out", out,
                   "--polish", str(phase1.POLISH_ITER), "--quiet", *extra, *COMMON,
                   "--wandb", W, "--wandb-group", f"round3_{tag}_{lv}",
                   "--wandb-job-type", "round3-exact", "--wandb-tags", "round-3", "phase-1", tag]
            jobs.append((f"{tag}/{name}", cmd, os.path.join(out, "stdout.log")))
phase1.pinned_pool(A, jobs)
print("TRAINING DONE", flush=True)

# 3. the control: Levenberg-Marquardt alone
run_phase1("lm", "--kinds", "noise", "--cpus", "p", "--wandb", W, "--wandb-tags", "round-3",
           "phase-1", "lm-only")
print("PHASE 1 DONE", flush=True)
