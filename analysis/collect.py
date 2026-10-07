"""Gather the summaries of all runs into one table.

    python analysis/collect.py [--runs-dir runs] [--out results.csv]
"""
import argparse
import csv
import glob
import json
import os

FIELDS = ["run", "n", "rank", "border", "d_min", "d_max", "seed", "best_val_loss", "best_epoch",
          "epsilon_at_best", "rho", "weight", "epochs_done", "max_epochs"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out", default="results.csv")
    a = ap.parse_args()
    rows = []
    for path in sorted(glob.glob(os.path.join(a.runs_dir, "*", "summary.json"))):
        with open(path) as fh:
            s = json.load(fh)
        s.update(run=os.path.basename(os.path.dirname(path)), d_min=s["degrees"][0],
                 d_max=s["degrees"][-1], border=int(s["border"]))
        rows.append({k: s[k] for k in FIELDS})
    with open(a.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    unfinished = sum(r["epochs_done"] < r["max_epochs"] for r in rows)
    print(f"{len(rows)} runs written to {a.out}" + (f" ({unfinished} not finished)" if unfinished else ""))


if __name__ == "__main__":
    main()
