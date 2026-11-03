"""Round 3, phase 1: per level, starts solved by each method (training also by training alone).
--csv FILE writes one row per run."""
import csv
import glob
import json
import os
import sys

D = "/home/marziali/MatMulNet/runs/round3-phase1"
LEVELS = ["noise_s0.4", "noise_s0.5", "noise_s0.6", "noise_s0.7", "noise_s0.8", "noise_s1"]
RHO_OK, W_MAX = 1e-10, 1e4

rows = []
for s in glob.glob(os.path.join(D, "exact", "*", "*", "summary.json")):
    j = json.load(open(s))
    if "polish_rho" not in j:
        continue
    tag, name = s.split(os.sep)[-3:-1]
    ok = j["polish_rho"] < RHO_OK and j["polish_weight"] < W_MAX
    rows.append(dict(method=tag, level=name.rsplit("_seed", 1)[0], start=name, solved=ok,
                     by_training=ok and j["stop"] == "rho", rho=j["polish_rho"],
                     weight=j["polish_weight"], train_rho=j["rho"], steps=j["steps_done"],
                     stop=j["stop"]))
for s in glob.glob(os.path.join(D, "lm", "*", "result.json")):
    j = json.load(open(s))
    rows.append(dict(method="lm", level=j["level"], start=j["start"], solved=j["solved"],
                     by_training="", rho=j["rho"], weight=j["weight"], train_rho="",
                     steps=j["iterations"], stop=j["status"]))

methods = sorted({r["method"] for r in rows}, key=lambda m: (m != "lm", m))
print(f"{'level':12s}" + "".join(f"{m:>22s}" for m in methods))
for lv in LEVELS:
    cells = []
    for m in methods:
        rs = [r for r in rows if r["method"] == m and r["level"] == lv]
        if not rs:
            cells.append(f"{'-':>22s}")
            continue
        txt = f"{sum(r['solved'] for r in rs)}/{len(rs)}"
        if m != "lm":
            txt += f" ({sum(r['by_training'] for r in rs)} train)"
        cells.append(f"{txt:>22s}")
    print(f"{lv:12s}" + "".join(cells))
if "--csv" in sys.argv:
    with open(sys.argv[sys.argv.index("--csv") + 1], "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["method"], r["level"], r["start"])))
