"""Round 3, phase 1: Levenberg-Marquardt continued from the unsolved points with a low residual
(rho < 0.11 at the end of phase 1), to tell exact decompositions from degenerations with growing
weight. Stages of 1000 iterations, at most 50,000 iterations or three hours; rho, W and the
factors are saved after every stage, and an interrupted run continues from its last stage.

    python results/4x4-round-3/lm_continue.py one lm noise_s0.6_seed14 """
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "experiments")
import phase1  # noqa: E402
from matmulnet import metrics  # noqa: E402
from refinement.refine import Problem, projected_lm  # noqa: E402

D = "/home/marziali/MatMulNet/runs/round3-phase1"
OUT = os.path.join(D, "lm_continue")
STAGE, MAX_ITER, MAX_SECONDS = 1000, 50_000, 3 * 3600


def targets():
    rows = csv.DictReader(open("results/4x4-round-3/results.csv"))
    return [(r["method"], r["start"]) for r in rows
            if r["solved"] != "True" and float(r["rho"]) < 0.11]


def start_file(method, name):
    if method == "lm":
        return os.path.join(D, "lm", name, "end.npz")
    return os.path.join(D, "exact", method, name, "polished.npz")


def jobs():
    out = []
    for method, name in targets():
        d = os.path.join(OUT, f"{method}_{name}")
        if os.path.exists(os.path.join(d, "result.json")):
            continue
        os.makedirs(d, exist_ok=True)
        cmd = [sys.executable, os.path.abspath(__file__), "one", method, name]
        out.append((f"lm-continue/{method}_{name}", cmd, os.path.join(d, "stdout.log")))
    return out


def one(method, name):
    d = os.path.join(OUT, f"{method}_{name}")
    trail, last = os.path.join(d, "trail.csv"), os.path.join(d, "end.npz")
    z = np.load(last if os.path.exists(last) else start_file(method, name))
    H, K, F = z["H"], z["K"], z["F"]
    prob = Problem(4, H.shape[1])
    x = prob.pack(H, K, F)
    done, spent = 0, 0.0
    if os.path.exists(trail):
        rows = list(csv.DictReader(open(trail)))
        if rows:
            done, spent = int(rows[-1]["iterations"]), float(rows[-1]["seconds"])
    else:
        with open(trail, "w") as fh:
            fh.write("iterations,rho,weight,status,seconds\n")
            fh.write(f"0,{metrics.rho(H, K, F, 4)!r},{metrics.weight(H, K, F)!r},start,0.0\n")
    status, rho, w = "max_iter", metrics.rho(H, K, F, 4), metrics.weight(H, K, F)
    while status == "max_iter" and done < MAX_ITER and spent < MAX_SECONDS:
        t0 = time.time()
        x, status, it = projected_lm(prob, x, np.inf, STAGE, max_seconds=MAX_SECONDS - spent)
        done, spent = done + it, spent + time.time() - t0
        H, K, F = prob.unpack(x)
        rho, w = metrics.rho(H, K, F, 4), metrics.weight(H, K, F)
        np.savez(last, H=H, K=K, F=F, n=4)
        with open(trail, "a") as fh:
            fh.write(f"{done},{rho!r},{w!r},{status},{spent:.1f}\n")
        print(f"{done} rho {rho:.6e} W {w:.6g} {status} {spent:.0f}s", flush=True)
    ref = np.load(os.path.join(D, "starts", "scheme.npz"))
    res = dict(method=method, start=name, rho=rho, weight=w, status=status, iterations=done,
               seconds=spent, solved=phase1.solved(rho, w),
               cancellation_level=metrics.cancellation_level(H, K, F, 4),
               terms_vs_scheme=phase1.same_terms(H, K, F, [ref[k] for k in "HKF"]))
    with open(os.path.join(d, "result.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    one(sys.argv[2], sys.argv[3])
