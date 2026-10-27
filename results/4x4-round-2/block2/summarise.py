"""Block 2: per rank (all lr together) and per rank and lr. -v lists every run; --csv FILE
writes one row per run."""
import csv
import glob
import json
import os
import statistics
import sys

D = "/home/marziali/MatMulNet/runs/round2-block2"


def runs():
    for s in sorted(glob.glob(os.path.join(D, "r*/lr*/seed*/summary.json"))):
        j = json.load(open(s))
        if "polish_rho" not in j:
            continue
        rank, lr, seed = s.split(os.sep)[-4:-1]
        j.update(rank_dir=rank, lr_dir=lr, seed_dir=seed,
                 solved=j["polish_rho"] < 1e-10 and j["polish_weight"] < 1e4)
        j["solved_by_training"] = j["solved"] and j["stop"] == "rho"
        j["candidate47"] = j["polish_rho"] < 1e-3 and j["polish_weight"] < 1e4
        yield j


def line(label, rs):
    rho = sorted(r["polish_rho"] for r in rs)
    st = [r["steps_done"] for r in rs]
    return (f"{label:14s} {len(rs):3d} runs  solved {sum(r['solved'] for r in rs):2d} "
            f"(training {sum(r['solved_by_training'] for r in rs):2d})  "
            f"rho<1e-3 & W<1e4 {sum(r['candidate47'] for r in rs):2d}  "
            f"rho min {rho[0]:.3g} median {statistics.median(rho):.3g}  "
            f"steps median {statistics.median(st):,.0f} max {max(st):,}")


rs = list(runs())
for rank in sorted({r["rank_dir"] for r in rs}):
    sub = [r for r in rs if r["rank_dir"] == rank]
    print(line(rank, sub))
    for lr in sorted({r["lr_dir"] for r in sub}, key=lambda x: float(x[2:])):
        print(line("  " + lr, [r for r in sub if r["lr_dir"] == lr]))
if "-v" in sys.argv:
    for r in rs:
        print(f"   {r['rank_dir']} {r['lr_dir']:7s} {r['seed_dir']:7s} {r['stop']:7s} "
              f"{r['steps_done']:>10,} train {r['rho']:.3g} polish {r['polish_rho']:.3g} "
              f"W {r['polish_weight']:.4g} 64rho2 {64 * r['rho'] ** 2:.2f}")
if "--csv" in sys.argv:
    keys = ["rank_dir", "lr_dir", "seed_dir", "solved", "solved_by_training", "candidate47", "stop",
            "steps_done", "rho", "weight", "polish_rho", "polish_weight", "polish_iterations",
            "seconds", "polish_seconds"]
    with open(sys.argv[sys.argv.index("--csv") + 1], "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "lr", "seed"] + keys[3:])
        for r in rs:
            w.writerow([r["rank_dir"][1:], r["lr_dir"][2:], r["seed_dir"][4:]]
                       + [r.get(k, "") for k in keys[3:]])
