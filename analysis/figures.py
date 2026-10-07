"""Figures of the paper.

    python analysis/figures.py rank       # rank_strip_linear, rank_strip_log
    python analysis/figures.py border     # brank_strip
    python analysis/figures.py frontier --r22 refinement_r22 --r23 refinement_r23
                                          # frontier_r22_r23

`rank` and `border` read results.csv (analysis/collect.py); `frontier` reads the frontier.csv
files written by refinement/refine.py. Figures are saved as PDF and PNG in --out.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, MUTED, GRID = "#1A1A1A", "#5B5B5B", "#DCDCDC"
MEAN, COOL, WARM = "#C0392B", "#5A7D9A", "#E08A3C"
C22, C23, FIT = "#2B6CB0", "#C05621", "#555555"

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": 9,
    "axes.labelsize": 9.5, "axes.titlesize": 9.5, "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.edgecolor": MUTED,
    "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "text.color": INK,
})


def losses(path, border):
    """{(rank, d_min, d_max): sorted best validation losses} for the 3x3 runs."""
    d = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if int(row["n"]) == 3 and int(row["border"]) == border:
                key = (int(row["rank"]), int(row["d_min"]), int(row["d_max"]))
                d.setdefault(key, []).append(float(row["best_val_loss"]))
    return {k: np.sort(v) for k, v in d.items()}


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("written", os.path.join(out, name + ".{pdf,png}"))


def strip(ax, x, v, color):
    """One dot per run, spread symmetrically around x, plus mean and standard deviation."""
    ax.plot(x + (np.arange(len(v)) - (len(v) - 1) / 2) * 0.085, v, "o", ms=4.2, color=color,
            mec="white", mew=0.5, alpha=0.9, zorder=3)
    m, s = v.mean(), v.std(ddof=1)
    low = s
    if ax.get_yscale() == "log" and m - s <= 0:
        # the lower end of the band is not on a logarithmic axis: draw it down to the frame
        ax.plot([x, x], [ax.get_ylim()[0], m], color=MEAN, lw=0.9, solid_capstyle="butt", zorder=2)
        low = 0.0
    ax.errorbar(x, m, yerr=[[low], [s]], fmt="s", ms=4.0, color=MEAN, ecolor=MEAN,
                elinewidth=0.9, capsize=2.6, capthick=0.9, zorder=5)


def style(ax):
    ax.grid(True, axis="y", which="major", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def dot(color):
    return plt.Line2D([], [], ls="none", marker="o", ms=4.2, color=color, mec="white", mew=0.5)


def rank(results, out):
    L = {k[0]: v for k, v in losses(results, 0).items()}
    ranks = sorted(L)
    for log in (False, True):
        fig, ax = plt.subplots(figsize=(3.7, 2.95))
        style(ax)
        ax.set_xlim(ranks[0] - 0.5, ranks[-1] + 0.5)
        if log:
            ax.set_yscale("log")
            ax.grid(True, axis="y", which="minor", color=GRID, lw=0.3, alpha=0.55)
            ax.set_ylim(2.5e-5, 2.2e-1)
        else:
            ax.set_ylim(-0.005, 0.105)
        for r in ranks:
            strip(ax, r, L[r], WARM if r >= 23 else COOL)
        ax.set_xlabel(r"rank parameter $r$")
        ax.set_ylabel("best validation loss" + (" (log scale)" if log else ""))
        ax.set_xticks(ranks)
        ax.legend([dot(COOL), dot(WARM), plt.Line2D([], [], marker="s", ms=4.0, color=MEAN, lw=0.9)],
                  [r"$r \leq 22$", r"$r \geq 23$", "mean $\\pm$ std"], frameon=False,
                  loc="lower left" if log else "upper right", handletextpad=0.5,
                  borderaxespad=0.4, labelspacing=0.3)
        save(fig, out, "rank_strip_log" if log else "rank_strip_linear")


def border(results, out):
    L = losses(results, 1)
    ranges = sorted({k[1:] for k in L}, reverse=True)
    ranks = sorted({k[0] for k in L})
    fig, axes = plt.subplots(1, len(ranges), figsize=(2.2 * len(ranges), 2.6), sharey=True,
                             squeeze=False)
    for ax, (lo, hi) in zip(axes[0], ranges):
        style(ax)
        ax.set_xlim(ranks[0] - 0.5, ranks[-1] + 0.5)
        ax.set_ylim(0.0, 0.128)
        for r in ranks:
            if (r, lo, hi) in L:
                strip(ax, r, L[(r, lo, hi)], COOL)
        ax.set_title(rf"$[d_{{\min}},d_{{\max}}]=[{lo},{hi}]$", pad=6)
        ax.set_xlabel(r"border rank parameter $r$")
        ax.set_xticks(ranks)
    axes[0][0].set_ylabel("best validation loss")
    axes[0][0].legend([dot(COOL), plt.Line2D([], [], marker="s", ms=4.0, color=MEAN, lw=0.9)],
                      ["single run", "mean $\\pm$ std"], frameon=False, loc="lower left",
                      handletextpad=0.5, borderaxespad=0.4, labelspacing=0.3)
    save(fig, out, "brank_strip")


def read_frontier(folder):
    with open(os.path.join(folder, "frontier.csv"), newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {"c": np.array([float(r["c"]) for r in rows]), "rho": np.array([float(r["rho"]) for r in rows]),
            "W": np.array([float(r["W"]) for r in rows]), "status": np.array([r["status"] for r in rows])}


def frontier(dir22, dir23, out):
    a, b = read_frontier(dir22), read_frontier(dir23)
    # power law of the r = 22 branch, fitted on the integer bounds c >= 5
    sel = (a["c"] >= 5) & (np.abs(a["c"] - np.round(a["c"])) < 1e-9)
    slope, icpt = np.polyfit(np.log10(a["W"][sel]), np.log10(a["rho"][sel]), 1)
    print(f"r = 22 branch: rho ~ W^{slope:.3f} on {sel.sum()} levels, "
          f"last point rho = {a['rho'][-1]:.2e} at W = {a['W'][-1]:.2e}")

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(5.4, 3.1), sharex=True,
                                 gridspec_kw={"height_ratios": [6, 1], "hspace": 0.10})
    for h in (ax, bx):
        h.set_xscale("log")
        h.set_yscale("log")
        h.grid(True, which="major", color=GRID, lw=0.6)
        h.set_axisbelow(True)
        h.spines["top"].set_visible(False)
        h.spines["right"].set_visible(False)

    xs = np.array([a["W"][sel].min() * 0.5, a["W"][sel].max() * 2.1])
    ax.plot(xs, 10 ** icpt * xs ** slope, "--", color=FIT, lw=1.1, zorder=2)
    ax.annotate(rf"$\rho \propto W^{{{slope:.2f}}}$", xy=(1.1e5, 10 ** icpt * 1.1e5 ** slope),
                xytext=(14, 16), textcoords="offset points", color=FIT, fontsize=9)
    ax.plot(a["W"], a["rho"], "-", color=C22, lw=0.9, alpha=0.5, zorder=3)
    ax.plot(a["W"], a["rho"], "o", ms=4.4, color=C22, mec="white", mew=0.6, zorder=5,
            label=r"$r=22$")
    ax.annotate(rf"$c={a['c'][-1]:g}$", xy=(a["W"][-1], a["rho"][-1]), xytext=(-1, -12),
                textcoords="offset points", ha="center", va="top", fontsize=8.5, color=C22)

    done = b["status"] == "machine_precision"
    ax.plot(b["W"][~done], b["rho"][~done], "-s", ms=4.4, color=C23, mec="white", mew=0.6,
            lw=0.9, zorder=6, label=r"$r=23$")
    ax.set_ylim(1.6e-7, 2.6)
    ax.set_ylabel(r"relative residual $\rho$")
    ax.legend(frameon=False, loc="upper right", handletextpad=0.5, borderaxespad=0.7,
              labelspacing=0.35)
    bx.set_ylim(8e-17, 4e-15)
    bx.set_yticks([1e-15])
    bx.set_xlabel(r"gauge-invariant weight  $W=\sum_{s}\|h_s\|\,\|k_s\|\,\|f_s\|$")
    bx.set_xlim(0.35, 2.4e6)

    if done.any():
        # the machine-precision point lives on the lower panel, below the axis break
        w, rho, c = b["W"][done][0], b["rho"][done][0], b["c"][done][0]
        dash = (0, (2.6, 2.4))
        ax.plot([w, w], [b["rho"][~done][-1] * 0.62, 1.6e-7], ls=dash, color=C23, lw=1.0, zorder=4)
        ax.text(w * 1.5, 1.1e-5, f"machine-precision\nnumerical solution\nat $c={c:g}$", color=C23,
                fontsize=8.5, ha="left", va="center", linespacing=1.3)
        bx.plot([w, w], [4e-15, rho * 1.55], ls=dash, color=C23, lw=1.0, zorder=4)
        bx.plot([w], [rho], marker="*", ms=12, color=C23, mec="white", mew=0.6, zorder=6)
        mant, expo = f"{rho:.1e}".split("e")
        bx.text(w * 1.5, rho, rf"$\rho={mant}\times10^{{{int(expo)}}}$", color=C23, fontsize=8.5,
                ha="left", va="center")

    mark = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=6, linestyle="none", color=MUTED,
                mec=MUTED, mew=1, clip_on=False)
    ax.plot([0], [0], transform=ax.transAxes, **mark)
    bx.plot([0], [1], transform=bx.transAxes, **mark)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(bottom=False)
    save(fig, out, "frontier_r22_r23")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["rank", "border", "frontier"])
    ap.add_argument("--results", default="results.csv")
    ap.add_argument("--r22", default="refinement_r22", help="refinement output for r = 22")
    ap.add_argument("--r23", default="refinement_r23", help="refinement output for r = 23")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    if a.what == "rank":
        rank(a.results, a.out)
    elif a.what == "border":
        border(a.results, a.out)
    else:
        frontier(a.r22, a.r23, a.out)


if __name__ == "__main__":
    main()
