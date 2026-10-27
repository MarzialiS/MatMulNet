"""Status of the long continuation: per run, the steps done and rho, 64 rho^2, W and lr now, against
the values at 10^7 steps (where block 2 stopped) and 24 hours of steps ago."""
import csv
import glob
import os

D = "/home/marziali/MatMulNet/runs/round2-block2-long"
DAY = int(24 * 3600 / 116e-6) // 10_000 * 10_000       # steps in about one day
print(f"{'run':20s} {'steps':>14s} {'rho@1e7':>8s} {'rho now':>8s} {'64rho2':>7s} "
      f"{'loss -%/day':>11s} {'W@1e7':>7s} {'W now':>8s} {'lr':>9s}")
for d in sorted(glob.glob(os.path.join(D, "r*/lr*/seed*"))):
    rows = list(csv.reader(open(os.path.join(d, "per_step.csv"))))[1:]
    by = {int(r[0]): r for r in rows}
    last = rows[-1]
    s, lr, loss, rho, w = int(last[0]), float(last[1]), float(last[2]), float(last[3]), float(last[4])
    ref = by[10_000_000]
    day = by.get(s - DAY)
    gain = 100 * (float(day[2]) - loss) / float(day[2]) if day else float("nan")
    print(f"{os.path.relpath(d, D):20s} {s:14,d} {float(ref[3]):8.4f} {rho:8.4f} {64 * rho * rho:7.3f} "
          f"{gain:11.3f} {float(ref[4]):7.0f} {w:8.0f} {lr:9.2e}")
