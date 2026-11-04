"""Round 3, ranks 46 and 49: per rank and learning rate, runs solved, residual, weight, steps and
the levels k = 64 rho^2 reached. --csv FILE writes one row per run."""
import collections
import csv
import glob
import json
import os
import statistics
import sys

D = "/home/marziali/MatMulNet/runs/round3-ranks"
RHO_OK, W_MAX = 1e-10, 1e4

rows = []
for s in glob.glob(os.path.join(D, "r*", "lr*", "seed*", "summary.json")):
    j = json.load(open(s))
    if "polish_rho" not in j:
        continue
    rank, lr, seed = s.split(os.sep)[-4:-1]
    ok = j["polish_rho"] < RHO_OK and j["polish_weight"] < W_MAX
    rows.append(dict(rank=int(rank[1:]), lr=lr[2:], seed=int(seed[4:]), solved=ok,
                     by_training=ok and j["stop"] == "rho", stop=j["stop"], steps=j["steps_done"],
                     train_rho=j["rho"], train_weight=j["weight"], rho=j["polish_rho"],
                     weight=j["polish_weight"], k=64 * j["rho"] ** 2, seconds=j["seconds"]))


def line(label, rs):
    rho = [r["rho"] for r in rs]
    stops = collections.Counter(r["stop"] for r in rs)
    return (f"{label:12s} {len(rs):4d} runs  solved {sum(r['solved'] for r in rs):3d} "
            f"(training {sum(r['by_training'] for r in rs):3d})  rho min {min(rho):.3g} "
            f"median {statistics.median(rho):.3g}  W median "
            f"{statistics.median(r['train_weight'] for r in rs):,.0f}  steps median "
            f"{statistics.median(r['steps'] for r in rs):,.0f}  stop {dict(stops)}")


for rank in sorted({r["rank"] for r in rows}):
    sub = [r for r in rows if r["rank"] == rank]
    print(line(f"r{rank}", sub))
    for lr in sorted({r["lr"] for r in sub}, key=float):
        print(line(f"  lr {lr}", [r for r in sub if r["lr"] == lr]))
    near = [r for r in sub if abs(r["k"] - round(r["k"])) < 0.05 * max(1, round(r["k"]))]
    print(f"  k: {sorted(collections.Counter(round(r['k']) for r in near).items())}, "
          f"near an integer {len(near)}/{len(sub)}")
if "--csv" in sys.argv:
    with open(sys.argv[sys.argv.index("--csv") + 1], "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["rank"], float(r["lr"]), r["seed"])))
