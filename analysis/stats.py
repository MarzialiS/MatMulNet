"""Summary tables and statistical tests on the best validation losses.

    python analysis/stats.py calibration   # 2x2, r = 6 vs 7
    python analysis/stats.py rank          # 3x3 rank sweep
    python analysis/stats.py border        # 3x3 border-rank sweep
    python analysis/stats.py precision     # dynamic range of the monomials in epsilon

The first three read results.csv, written by analysis/collect.py.
"""
import argparse
import csv
import itertools
import math

import numpy as np
from scipy import stats

TAU_RANK, TAU_CALIBRATION = 1e-3, 1e-12


def load(path, **where):
    out = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if all(int(row[k]) == v for k, v in where.items()):
                out.append(row)
    return out


def groups(rows, key, value="best_val_loss"):
    d = {}
    for row in rows:
        d.setdefault(int(row[key]), []).append(float(row[value]))
    return {k: np.array(v) for k, v in sorted(d.items())}


def jonckheere_terpstra(samples):
    """Ordered alternative: the loss decreases along the list of samples. J counts the pairs,
    taken from two different samples, in which the later sample has the smaller value; Z is its
    excess over the null expectation in units of the null standard deviation."""
    J = 0.0
    for a, b in itertools.combinations(samples, 2):
        for x in a:
            J += np.sum(b < x) + 0.5 * np.sum(b == x)
    sizes = [len(s) for s in samples]
    N = sum(sizes)
    mean = (N * N - sum(k * k for k in sizes)) / 4
    var = (N * N * (2 * N + 3) - sum(k * k * (2 * k + 3) for k in sizes)) / 72
    z = (J - mean) / math.sqrt(var)
    return J, z, stats.norm.sf(z)


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted, running = np.empty(len(pvalues)), 0.0
    for i, idx in enumerate(order):
        running = max(running, (len(pvalues) - i) * pvalues[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def welch_table(L, pairs):
    """One-tailed Welch t-tests (the loss is larger at the smaller rank) with Holm correction
    over the listed pairs, and Cohen's d with the pooled standard deviation."""
    rows = []
    for a, b in pairs:
        x, y = L[a], L[b]
        res = stats.ttest_ind(x, y, equal_var=False, alternative="greater")
        d = (x.mean() - y.mean()) / math.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
        rows.append((a, b, res.statistic, res.df, res.pvalue, d))
    adj = holm([r[4] for r in rows])
    print("   pair        t     df        p    p_Holm     d")
    for (a, b, t, df, p, d), ph in zip(rows, adj):
        print(f"  {a} vs {b}  {t:6.2f}  {df:5.1f}  {p:.1e}  {ph:.1e}  {d:5.2f}")


def summary(L, tau=None, R=None):
    head = f"  {'r':>3} {'n':>2} {'mean':>10} {'std':>10} {'median':>10} {'min':>10}"
    print(head + ("  rho median   rho min" if R else "") + ("  successes" if tau else ""))
    for r, v in L.items():
        line = f"  {r:>3} {len(v):>2} {v.mean():10.3e} {v.std(ddof=1):10.3e} {np.median(v):10.3e} {v.min():10.3e}"
        if R:
            line += f"  {np.median(R[r]):10.3f} {R[r].min():9.3f}"
        if tau:
            line += f"  {int((v < tau).sum())}/{len(v)}"
        print(line)


def calibration(path):
    L = groups(load(path, n=2, border=0), "rank")
    print(f"2x2 multiplication, best validation loss (success: below {TAU_CALIBRATION:g})")
    summary(L, TAU_CALIBRATION)
    for r, v in L.items():
        print(f"  r = {r}: range [{v.min():.3e}, {v.max():.3e}]")


def rank(path):
    rows = load(path, n=3, border=0)
    L, R = groups(rows, "rank"), groups(rows, "rank", "rho")
    ranks = list(L)
    print(f"3x3 multiplication, best validation loss (success: below {TAU_RANK:g})")
    summary(L, TAU_RANK, R)

    allv = np.sort(np.concatenate(list(L.values())))
    ok, ko = allv[allv < TAU_RANK], allv[allv >= TAU_RANK]
    if len(ok) and len(ko):
        print(f"\nlargest loss below the threshold {ok.max():.2e}, smallest above {ko.min():.2e} "
              f"(factor {ko.min() / ok.max():.0f})")

    low = [r for r in ranks if r <= 21]
    high = [r for r in ranks if r >= 22]
    s_low, n_low = sum(int((L[r] < TAU_RANK).sum()) for r in low), sum(len(L[r]) for r in low)
    s_high, n_high = sum(int((L[r] < TAU_RANK).sum()) for r in high), sum(len(L[r]) for r in high)
    p = stats.fisher_exact([[s_low, n_low - s_low], [s_high, n_high - s_high]])[1]
    print(f"\nFisher's exact test, r <= 21 vs r >= 22: {s_low}/{n_low} vs {s_high}/{n_high}, p = {p:.1e}")

    J, z, p = jonckheere_terpstra([L[r] for r in ranks])
    print(f"Jonckheere-Terpstra, r = {ranks[0]}..{ranks[-1]}: Z = {z:.2f}, p = {p:.1e}")

    print("\nWelch t-tests between consecutive ranks")
    welch_table(L, list(zip(ranks[:-1], ranks[1:])))

    fit = [r for r in ranks if r <= 23]
    slope, icpt, rv, *_ = stats.linregress(fit, [math.log(L[r].mean()) for r in fit])
    print(f"\nmean loss, r = {fit[0]}..{fit[-1]}: factor {math.exp(slope):.2f} per unit of rank "
          f"(R^2 = {rv * rv:.2f})")

    failed = [(r, x) for r in high for x in R[r][L[r] >= TAU_RANK]]
    if failed:
        print("\nresidual rho of the runs above the threshold at r >= 22")
        for r in high:
            v = np.sort([x for q, x in failed if q == r])
            print(f"  r = {r}: " + " ".join(f"{x:.4f}" for x in v))


def border(path):
    rows = load(path, n=3, border=1)
    ranges = sorted({(int(r["d_min"]), int(r["d_max"])) for r in rows}, reverse=True)
    cells = {rg: groups([r for r in rows if (int(r["d_min"]), int(r["d_max"])) == rg], "rank")
             for rg in ranges}
    for rg in ranges:
        print(f"exponent range [{rg[0]},{rg[1]}], best validation loss")
        summary(cells[rg])
        J, z, p = jonckheere_terpstra(list(cells[rg].values()))
        print(f"  Jonckheere-Terpstra: Z = {z:.2f}, p = {p:.1e}\n")

    ranks = sorted(set.intersection(*(set(c) for c in cells.values())))
    y = np.array([[cells[rg][r] for rg in ranges] for r in ranks])       # rank x range x run
    a, b, k = y.shape
    gm = y.mean()
    ss_r = b * k * ((y.mean(axis=(1, 2)) - gm) ** 2).sum()
    ss_d = a * k * ((y.mean(axis=(0, 2)) - gm) ** 2).sum()
    ss_cell = k * ((y.mean(axis=2) - gm) ** 2).sum()
    ss_e = ((y - y.mean(axis=2, keepdims=True)) ** 2).sum()
    df_e = a * b * (k - 1)
    print("two-way ANOVA (rank x exponent range)")
    print("  source            SS   df          MS        F        p")
    for name, ss, df in (("rank", ss_r, a - 1), ("exponent range", ss_d, b - 1),
                         ("interaction", ss_cell - ss_r - ss_d, (a - 1) * (b - 1))):
        F = (ss / df) / (ss_e / df_e)
        print(f"  {name:<15} {ss:.5f}  {df:3d}  {ss / df:.3e}  {F:7.2f}  {stats.f.sf(F, df, df_e):.1e}")
    print(f"  {'residual':<15} {ss_e:.5f}  {df_e:3d}  {ss_e / df_e:.3e}")

    print("\nmean loss per rank and exponent range")
    print("    r  " + "  ".join(f"[{p},{q}]".rjust(8) for p, q in ranges))
    for i, r in enumerate(ranks):
        print(f"  {r:3d}  " + "  ".join(f"{v:8.4f}" for v in y[i].mean(axis=1)))

    main_range = ranges[0]
    print(f"\nWelch t-tests between consecutive ranks, range [{main_range[0]},{main_range[1]}]")
    welch_table(cells[main_range], list(zip(ranks[:-1], ranks[1:])))


def precision(eps_values=(1e-3, 1e-1), ranges=((-2, 4), (-3, 5), (-4, 6))):
    for eps in eps_values:
        print(f"epsilon = {eps:g}")
        print("  range     largest monomial  smallest monomial  log10(ratio)")
        for lo, hi in ranges:
            print(f"  [{lo},{hi}]   {eps ** lo:16.0e}  {eps ** hi:17.0e}  "
                  f"{(hi - lo) * -math.log10(eps):12.0f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["calibration", "rank", "border", "precision"])
    ap.add_argument("--results", default="results.csv")
    a = ap.parse_args()
    if a.what == "precision":
        precision()
    else:
        {"calibration": calibration, "rank": rank, "border": border}[a.what](a.results)


if __name__ == "__main__":
    main()
