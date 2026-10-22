# 4×4 round 2 — block 1: choice of the optimiser configuration at σ = 0.6

Run 22 October 2026 (19:15–20:03) on the 8 P-cores of the i9-13900K. Launcher:
`run_selection.py` (run from the repository root); raw runs in
`runs/round2-phase1/exact/S-<config>/` on the machine; W&B job type `select-s0.6`, tag `selection`.

## Rule (fixed before the runs)

- Starts: `noise_s0.6`, seeds 1–10 (LM alone, 1000 iterations: 8/10).
- Configurations: the 19 of phase 1b without the discrete pressures (proto, A, B, C).
- Exact loss, at most 10⁶ Adam steps, stop at ρ < 1e-8 or on a stall, then at most 200 LM
  iterations.
- Reliability: starts solved by training alone (ρ < 1e-8 before the polish, ρ < 1e-10 and
  W < 1e4 after). Same reliability: at most one start fewer than the best configuration.
  Among those, the lowest median number of Adam steps over the solved starts.

## Result

| configuration | solved by training | steps of the solved starts |
|---|---|---|
| A-lr1e-2 (plateau ×0.5 / 10⁴, 1e-2 → 1e-8) | 2/10 (seeds 3, 5) | 18,700, 36,600 (median 27,650) |
| A-lr3e-3 | 1/10 (seed 3) | 23,500 |
| A-lr3e-4 | 1/10 (seed 3) | 39,500 |
| A-lr1e-3, B-lr1e-3-min1e-6, B-lr1e-3-min1e-10 | 1/10 (seed 3) | 81,900 (same trajectory) |
| B-lr1e-3-cos, B-lr1e-3-const | 0/10 (seed 3 solved only with the polish) | – |
| proto, A-lr1e-4, B-lr1e-4-*, C-* (all with initial lr 1e-4) | 0/10 | – |

By the rule the configurations with at least 1/10 tie, and **A-lr3e-3** has the lowest median
(23,500 steps). The medians rest on one or two runs, and seed 3 is the only start solved by
more than one configuration.

Unsolved runs: 169 of 190 end with 64ρ² within 5% of an integer k = 1…5 (k = 1: 24, 2: 46,
3: 43, 4: 33, 5: 23), the spurious minima of phase 1b. With initial lr 1e-4, seed 3 reaches
the step budget at ρ ≈ 2·10⁻³ (3.8·10⁻⁴ after the polish) with W ≈ 2,200, still decreasing.

Every run: `results.csv`; per run: `table.txt`.
