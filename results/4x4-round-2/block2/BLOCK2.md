# 4×4 round 2 — block 2: 47 and 48 terms from random starts

Run 26–27 October 2026 (22:06–01:28) on the 8 P-cores of the i9-13900K. Launcher:
`run_block2.py`, summary: `summarise.py` (both run from the repository root); raw runs in
`runs/round2-block2/` on the machine; W&B job type `block2-scratch`, tag `block-2`.

## Protocol (fixed before the runs)

- r = 47 and 48; factors drawn from N(0, 1) (seeds 1–10), no link to the known scheme.
- Exact loss; Adam, plateau ×0.5 / patience 10⁴ steps down to 1e-8, initial learning rate
  3e-4, 1e-3, 3e-3, 1e-2 (the configurations that solved at least one start at σ = 0.6 by
  training alone, `../selection/SELECTION.md`); 10 runs per learning rate and rank, 80 runs.
- At most 10⁷ steps, stop at ρ < 1e-8 or on a stall, then at most 200 LM iterations.
- Solved: ρ < 1e-10 and W < 1e4 after the polish. Candidate at 47: ρ < 1e-3 and W < 1e4.
- Results reported per rank, all learning rates together.

## Result

| r | runs | solved | candidates (ρ < 1e-3, W < 1e4) | ρ min | ρ median | 64ρ² (k: runs) |
|---|---|---|---|---|---|---|
| 47 | 40 | 0 | 0 | 0.280 | 0.331 | 5: 7, 6: 8, 7: 8, 8: 11, 9: 6 |
| 48 | 40 | 0 | 0 | 0.177 | 0.280 | 2: 1, 3: 4, 4: 5, 5: 12, 6: 5, 7: 5, 8: 4, 9: 3, 10: 1 |

Per learning rate: `table.txt`; every run: `results.csv`.

- No run solved and no candidate at 47: by the criteria agreed beforehand, the 3×3 case stays
  the main result.
- All runs end in the spurious minima of phase 1b: 79 of 80 with 64ρ² within 5% of an
  integer k (the 48 terms approximate M₄ minus k rank-one tensors), with W in the thousands
  (median 5,400–5,700 at the end of training, 15,000 after the polish; the scheme has 210).
  From random starts k is larger than from the perturbed scheme (5–9 at 47, 2–10 at 48).
- The minima at 48 are lower than at 47 (median k 5 against 7; Mann–Whitney on ρ,
  p = 4·10⁻⁴). This is a gradual difference between two failures, not a contrast between a
  rank that is found and one that is not; round 1 (sampled loss, 7 runs per rank) did not
  separate 47 from 48 (median ρ 0.270 and 0.279).

## Note on the stopping rule

No run stopped on a stall: all 80 used the 10⁷ steps (25.9 CPU-hours). The plateau scheduler
follows `torch.optim.lr_scheduler.ReduceLROnPlateau`, which skips a reduction smaller than
1e-8; halving from these initial rates, the learning rate stops at 1.5–2·10⁻⁸ and never
reaches the floor 1e-8, and the stall rule waits for the floor. The outcome is not affected —
between 10⁶ and 10⁷ steps ρ falls by about 0.1% while W grows by about 20% — but the runs took
the whole budget. The same holds for the phase-1b and block-1 configurations with floor 1e-8.
Fixed afterwards in `matmulnet/schedule.py` (every reduction is applied down to `--lr-min`; see
`../PHASE1B.md`, "Correction"); the runs of this block were made with the old rule.
