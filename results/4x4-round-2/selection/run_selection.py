"""Block 1: choice of the optimiser configuration at sigma = 0.6 (exact loss, 10^6 steps).
The 19 configurations of phase 1b without the discrete pressures (proto, A, B, C), seeds 1-10,
all 190 runs in one pool on the 8 P-cores. Run from the worktree root."""
import json
import os
import sys

sys.path.insert(0, "experiments")
import phase1  # noqa: E402

D = "/home/marziali/MatMulNet/runs/round2-phase1"
W = "sara-marziali-university-of-siena/matmulnet-4x4"
STEPS = 1_000_000

CONFIGS = {"proto": ["--lr", "1e-4", "--lr-min", "1e-6", "--patience", "5000"]}
for lr in ["1e-4", "3e-4", "1e-3", "3e-3", "1e-2"]:
    CONFIGS[f"A-lr{lr}"] = ["--lr", lr, "--lr-min", "1e-8", "--patience", "10000"]
for lr in ["1e-4", "1e-3"]:
    for lmin in ["1e-6", "1e-10"]:
        CONFIGS[f"B-lr{lr}-min{lmin}"] = ["--lr", lr, "--lr-min", lmin, "--patience", "10000"]
    CONFIGS[f"B-lr{lr}-cos"] = ["--schedule", "cosine", "--lr", lr, "--lr-min", "1e-8"]
    CONFIGS[f"B-lr{lr}-const"] = ["--schedule", "constant", "--lr", lr]
for f in ["0.5", "0.1"]:
    for p in ["1000", "10000", "100000"]:
        if (f, p) != ("0.5", "10000"):
            CONFIGS[f"C-f{f}-p{p}"] = ["--lr", "1e-4", "--lr-min", "1e-8", "--lr-factor", f,
                                       "--patience", p]
assert len(CONFIGS) == 19


class A:
    cpus = "p"
    jobs = None


jobs = []
for tag, extra in CONFIGS.items():
    stag = "S-" + tag
    for seed in range(1, 11):
        name = f"noise_s0.6_seed{seed}"
        out = os.path.join(D, "exact", stag, name)
        summary = os.path.join(out, "summary.json")
        if os.path.exists(summary) and "polish_rho" in json.load(open(summary)):
            continue
        os.makedirs(out, exist_ok=True)
        cmd = [sys.executable, "-m", "matmulnet.exact", "--n", "4", "--rank", "48",
               "--init-from", os.path.join(D, "starts", name + ".npz"), "--out", out,
               "--polish", str(phase1.POLISH_ITER), "--quiet", "--steps", str(STEPS), *extra,
               "--wandb", W, "--wandb-group", f"select_{tag}_noise_s0.6",
               "--wandb-job-type", "select-s0.6", "--wandb-tags", "round-2", "selection", tag]
        jobs.append((f"{stag}/{name}", cmd, os.path.join(out, "stdout.log")))

phase1.pinned_pool(A, jobs)
print("BLOCK 1 DONE", flush=True)
