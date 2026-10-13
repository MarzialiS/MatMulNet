"""Collect the results of the round into report/data.json.

    python results/4x4-round-1/report/gather.py      (from the root of the repository)
"""
import csv, glob, gzip, io, json, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                         # results/4x4-round-1
sys.path.insert(0, os.path.dirname(os.path.dirname(ROOT)))
from matmulnet import metrics  # noqa: E402
os.chdir(ROOT)
D = {}
def summ(p): return json.load(open(p))
# rank
D["rank"] = [dict(r=s["rank"], seed=s["seed"], loss=s["best_val_loss"], rho=s["rho"], W=s["weight"], sec=s["seconds"])
             for s in map(summ, sorted(glob.glob("training/rank_n4_r*_seed*/summary.json")))]
D["border"] = []
for p in sorted(glob.glob("training/border_n4_*/summary.json")):
    s = summ(p); L = json.load(open(p.replace("summary.json", "best.laurent.json")))
    tr = [r for r in L["at_epsilon"] if r["epsilon"] == L["epsilon_trained"]][0]
    D["border"].append(dict(r=s["rank"], seed=s["seed"], loss=s["best_val_loss"], eps=s["epsilon_at_best"], epoch=s["best_epoch"],
        last=s["last_val_loss"], rho_tr=tr["rho"], neg=L["negative_part"], t0=L["constant_term_distance"],
        at=[(r["epsilon"], r["rho"]) for r in L["at_epsilon"] if r["epsilon"] != L["epsilon_trained"]], sec=s["seconds"]))
def curve(path, xcol, idx):
    rows = list(csv.DictReader(io.TextIOWrapper(gzip.open(path))))
    return [(float(rows[i][xcol]), float(rows[i]["val_loss"])) for i in idx]
idx = sorted(set(int(round(v)) - 1 for v in np.logspace(0, 4, 70)))          # rows are every 100 epochs
D["rank_curves"] = {}
for r in (45, 46, 47, 48, 49):
    cs = np.array([[v for _, v in curve(p, "epoch", idx)] for p in sorted(glob.glob(f"training/rank_n4_r{r}_seed*/per_epoch.csv.gz"))])
    D["rank_curves"][r] = [( (i + 1) * 100, float(m)) for i, m in zip(idx, np.median(cs, 0))]
for r in (51, 54, 60, 64):
    D["rank_curves"][r] = [((i + 1) * 100, v) for i, (_, v) in zip(idx, curve(f"training/rank_n4_r{r}_seed1/per_epoch.csv.gz", "epoch", idx))]
idl = sorted(set(int(round(v)) for v in np.linspace(0, 9999, 90)))
D["border_curves"] = {}
for r in (44, 45, 46, 47):
    cs = [curve(p, "epsilon", idl) for p in sorted(glob.glob(f"training/border_n4_r{r}_*/per_epoch.csv.gz"))]
    eps = [e for e, _ in cs[0]]
    D["border_curves"][r] = list(zip(eps, [float(x) for x in np.median(np.array([[v for _, v in c] for c in cs]), 0)]))
D["smirnov_ctrl"] = curve("training/control_smirnov46_train/per_epoch.csv.gz", "epsilon", idl)
z = np.load("known_schemes/smirnov46.npz"); co = [z[k] for k in ("H_coeff", "K_coeff", "F_coeff")]; deg = list(z["degrees"])
D["smirnov_exact"] = []
for e in np.logspace(-1, -6, 26):
    H, K, F = (metrics.evaluate(c, deg, e) for c in co)
    rho = metrics.rho(H, K, F, 4); D["smirnov_exact"].append((float(e), rho, 4 * rho * rho / 9))
# refinement
def frontier(d):
    return [dict(c=float(r["c"]), rho=float(r["rho"]), W=float(r["W"]), status=r["status"], it=int(r["iterations"])) for r in csv.DictReader(open(d + "/frontier.csv"))]
D["refine"] = []
for d in sorted(glob.glob("refinement/direct_rank_*")):
    name = os.path.basename(d)
    r, seed = (int(x) for x in re.search(r"r(\d+)_s(?:eed)?(\d+)", name).groups())
    s = summ(f"training/rank_n4_r{r}_seed{seed}/summary.json")
    D["refine"].append(dict(r=r, seed=seed, start=dict(rho=s["rho"], W=s["weight"]), levels=frontier(d)))
D["chain"] = {k: frontier(f"refinement/chain_{k}_s1e-2") for k in ("strassen49", "rational48")}
D["direct_ctrl"] = {k: frontier(f"refinement/direct_{k}_s1e-2") for k in ("strassen49", "rational48")}
D["ctrl_train"] = [dict(name=os.path.basename(os.path.dirname(p)), **{k: summ(p)[k] for k in ("best_val_loss", "rho", "last_val_loss", "last_rho")})
                   for p in sorted(glob.glob("training/control_*49*/summary.json") + glob.glob("training/control_*48*/summary.json"))]
s = summ("training/control_smirnov46_train/summary.json")
D["smirnov_sum"] = {k: s[k] for k in ("best_val_loss", "best_epoch", "epsilon_at_best", "last_val_loss", "rho")}
json.dump(D, open(os.path.join(HERE, "data.json"), "w"))
