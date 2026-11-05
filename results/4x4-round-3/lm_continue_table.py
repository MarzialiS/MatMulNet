"""LM continuation: one row per run, with rho and W before and after and the slope of log rho
against log W over the second half of the run. Writes lm_continue_results.csv next to this file."""
import csv
import glob
import json
import math
import os

D = "/home/marziali/MatMulNet/runs/round3-phase1/lm_continue"
HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "lm_continue_results.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["method", "start", "solved", "status", "iterations", "seconds", "rho_before",
                "rho_after", "weight_before", "weight_after", "slope_second_half",
                "terms_vs_scheme"])
    for d in sorted(glob.glob(os.path.join(D, "*"))):
        rows = list(csv.DictReader(open(os.path.join(d, "trail.csv"))))
        a, m, b = rows[0], rows[len(rows) // 2], rows[-1]
        slope = math.log(float(b["rho"]) / float(m["rho"])) / \
            math.log(float(b["weight"]) / float(m["weight"]))
        j = json.load(open(os.path.join(d, "result.json")))
        w.writerow([j["method"], j["start"], j["solved"], j["status"], j["iterations"],
                    round(j["seconds"]), a["rho"], b["rho"], a["weight"], b["weight"],
                    round(slope, 3), j["terms_vs_scheme"]])
