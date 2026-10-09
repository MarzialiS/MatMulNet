"""Check whether an eps-parametrised decomposition converges to M_n as eps -> 0.

    python analysis/laurent.py runs/border_n4_r46_d-1_2_scaled_seed1/best.npz

With the coefficients held fixed, the represented tensor is a Laurent polynomial in eps,
T(eps) = sum_d eps^d T_d. The family converges to M_n if and only if T_d = 0 for every d < 0
and T_0 = M_n. The script prints the norm of every T_d relative to ||M_n||, the distance
||T_0 - M_n|| / ||M_n||, and, for a decreasing sequence of values of eps, the residual rho,
the weight W and the double-precision cancellation level of the factors evaluated at eps.
The same numbers are written to <file>.laurent.json.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from matmulnet import metrics  # noqa: E402

EPS = [1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 1e-4, 1e-5, 1e-6]


def analyse(path, eps_values=EPS):
    z = np.load(path)
    coeff = [np.asarray(z[k], dtype=np.float64) for k in ("H_coeff", "K_coeff", "F_coeff")]
    deg = [int(d) for d in z["degrees"]]
    n = int(round(np.sqrt(coeff[0].shape[0])))
    M = metrics.matmul_tensor(n)
    norm = float(np.linalg.norm(M))
    lau = metrics.laurent(*coeff, deg)
    out = dict(file=path, n=n, rank=coeff[0].shape[1], degrees=deg,
               epsilon_trained=float(z["epsilon"]) if "epsilon" in z.files else None,
               coefficient_norms={d: float(np.linalg.norm(t)) / norm for d, t in lau.items()},
               negative_part=float(np.sqrt(sum(np.linalg.norm(t) ** 2 for d, t in lau.items()
                                               if d < 0))) / norm,
               constant_term_distance=float(np.linalg.norm(lau[0] - M)) / norm, at_epsilon=[])
    values = sorted(set(eps_values) | ({out["epsilon_trained"]} - {None}), reverse=True)
    for eps in values:
        H, K, F = (metrics.evaluate(c, deg, eps) for c in coeff)
        out["at_epsilon"].append(dict(epsilon=eps, rho=metrics.rho(H, K, F, n),
                                      weight=metrics.weight(H, K, F),
                                      cancellation_level=metrics.cancellation_level(H, K, F, n)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help=".npz with H_coeff, K_coeff, F_coeff and degrees")
    ap.add_argument("--eps", type=float, nargs="+", default=EPS)
    a = ap.parse_args()
    for path in a.files:
        out = analyse(path, a.eps)
        print(f"{path}: n = {out['n']}, r = {out['rank']}, exponents {out['degrees'][0]}..{out['degrees'][-1]}")
        for d, v in out["coefficient_norms"].items():
            print(f"  ||T_{d}|| / ||M|| = {v:.3e}")
        print(f"  negative part {out['negative_part']:.3e}   ||T_0 - M|| / ||M|| = "
              f"{out['constant_term_distance']:.3e}")
        for row in out["at_epsilon"]:
            mark = "  (training)" if row["epsilon"] == out["epsilon_trained"] else ""
            print(f"  eps = {row['epsilon']:.3e}  rho = {row['rho']:.3e}  W = {row['weight']:.3e}  "
                  f"cancellation {row['cancellation_level']:.1e}{mark}")
        with open(os.path.splitext(path)[0] + ".laurent.json", "w") as fh:
            json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
