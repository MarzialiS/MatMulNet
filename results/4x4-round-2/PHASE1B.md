# 4×4 round 2 — phase 1b: recovering the 48-term decomposition with the exact loss

Run 14 October 2026 on the i9-13900K (8 P-cores, one single-threaded process per core).
Code: branch `round2-phase1b` (`matmulnet/exact.py`, command `exact` of
`experiments/phase1.py`). Raw runs: `runs/round2-phase1/exact/` on the machine (not in git).
Every run is on W&B, project `sara-marziali-university-of-siena/matmulnet-4x4`, tags `round-2`,
`phase-1b` and the configuration's tag, job type `phase1b-exact`.

## Question

Phase 1 (`PHASE1.md`) left training useful only up to t = 0.3 and σ = 0.7 (1/5). Same task —
recover an exact 48-term decomposition of M₄ from a perturbed copy of the rational scheme of
Dumas–Pernet–Sedoglavic — now training on the exact loss, raising t and σ once a configuration
converges.

## Set-up

- **Exact loss**: `‖T − M₄‖²_F / (9 n²)` = 4ρ²/9, the expectation of the per-entry squared error
  that the sampled protocol estimates (entries of A, B uniform in [−1, 1]); no training or
  validation sets, every step a full-batch Adam step (gradient norm clipped at 10, as before).
  ≈ 116 µs per step on one P-core.
- **Budget per run**: at most 2·10⁶ steps; stop when ρ < 1e-8, or when the best loss has not
  improved by a relative 1e-3 for 10⁵ steps while the learning rate cannot decrease any more;
  then at most 200 Levenberg–Marquardt iterations (the polish of phase 1).
- **Solved**: ρ < 1e-10 with W < 1e4 after the polish. "Training alone" (`t` in the tables):
  ρ < 1e-8 before the polish.
- **Starts**: 10 per level (seeds 1–10), built as in phase 1. A level counts only if LM alone
  (1000 iterations) solves at most 2 of them.
- **Plan**: A — initial learning rate; B — final learning rate, cosine and constant as
  references; C — plateau factor and patience; D, if all fail — pressures towards discrete
  solutions. Move to larger t, σ once a configuration solves ≥ 5/10 at a level.

## Results

LM-alone control: σ = 0.6 is solved by LM alone in 8/10 starts, so the noise level used is
σ = 0.7 (0/10), together with t = 0.4 (0/10).

| stage | configurations (20 runs each: 10 at σ = 0.7, 10 at t = 0.4) | solved |
|---|---|---|
| baseline | protocol on the exact loss: Adam 1e-4, plateau ×0.5 / patience 5,000 steps, down to 1e-6 | 0/20 |
| A | plateau ×0.5 / 10⁴ steps down to 1e-8, initial lr 1e-4, 3e-4, 1e-3, 3e-3, 1e-2 | 0/100 |
| B | initial lr 1e-4 and 1e-3: final lr 1e-6, 1e-10; cosine to 1e-8; constant | 0/160 |
| C | lr 1e-4 → 1e-8, plateau factor 0.5, 0.1 × patience 10³, 10⁴, 10⁵ steps | 0/100 |
| D | cosine 1e-3 → 1e-6, 2·10⁶ steps, with: grid penalty λ = 1e-4, 1e-3, 1e-2; L1 + balanced terms λ = 0.03, 0.1, 0.3; progressive snapping and freezing, alone and with the grid penalty 1e-3 | 0/160 |

**0/540** in total (27 configurations). Full table: `phase1b/table.txt`; every run:
`phase1b/results.csv`.

Check that the exact loss is a valid training method where the task is within reach (t = 0.3,
seeds 1–5): protocol on the exact loss 4/5 and lr 1e-3 4/5, all four by training alone
(ρ < 1e-8 before the polish); LM alone 1/5; the sampled protocol of phase 1 5/5.

## What the runs show

1. **All runs end in the same family of minima, whatever the schedule.** Of the 380 runs of
   the baseline and stages A–C, 369 end with 64 ρ² within 5% of an integer k ∈ {1, …, 6}
   (k = 4: 134 runs, 3: 125, 5: 74, 2: 25, 6: 10, 1: 1). The residual R = T − M₄ has, in each of its
   three flattenings, exactly k singular values equal to 1 and the others below 0.1: the 48 terms
   approximate M₄ minus k orthonormal rank-one tensors — "M₄ minus k entries" up to the
   symmetries of M₄, which is why the residual looks spread out in the standard basis. They
   approach that tensor with growing weight (W median 3,200 at the end of training, against
   210 for the scheme), a border-rank-like degeneration: in the limit, 48 + k terms.
2. **The minimum is chosen early.** 64 ρ² is within 10% of its final integer after a median of
   50,000 steps at lr 1e-4, 19,000 at 1e-3, 11,000 at 1e-2; what the schedule does afterwards
   (final learning rate down to 1e-10, plateau factor and patience, cosine, constant) only moves
   the run along the degenerate valley. The polish from these points does not leave them either
   (W of the polished points: median 16,600).
3. **The pressures towards discrete solutions do not help**, because they act on points that
   are already in those minima, whose coefficients are far from the grid (W 700–2,300 against
   210):
   - grid penalty (0, ±1/8, ±1/4, ±1/2, ±1): ρ unchanged (0.25–0.32), the points do not
     approach the grid (ρ of the rounded point 2.6–5);
   - L1 with balanced terms: λ = 0.03 changes nothing (ρ 0.22), λ = 0.1 and 0.3 shrink the
     decomposition (W 67, 25) and ρ rises to 0.47, 0.78;
   - snapping and freezing: 77–87% of the coefficients end frozen, ρ 1.0–1.4 before the
     polish, which brings it back to the same minima (0.25).
4. The exact loss changes the dynamics with respect to the sampled one only in the noise:
   where the task is within reach (t = 0.3) both converge; at t = 0.4 and σ = 0.7 both end in
   the same minima (phase 1: ρ 0.18–0.29, the same values).

## Conclusion

No configuration of learning rate (initial 1e-4 to 1e-2, final down to 1e-10) or scheduler
(plateau with factors 0.5, 0.1 and patience 10³–10⁵ steps, cosine, constant), and none of the
three discrete pressures tried, makes training useful beyond t = 0.3 / σ = 0.6. The ladder of
levels stops at its first step (t = 0.4, σ = 0.7). The obstacle is a precise family of
spurious minima — 48 terms fitting M₄ minus k rank-one tensors, with diverging weight — that
the run enters in its first 10⁴–10⁵ steps.

Directions suggested by the diagnosis, not tried yet:

- **Keep the weight bounded during training** (a box |θ| ≤ c or a bound on W enforced by
  projection, as in `refinement/`): the minima are reached only with W in the thousands, the
  scheme has W = 210. Weight decay (phase 1) lowered W without changing ρ, but a hard bound
  removes the degenerate valley instead of tilting it.
- **Discrete pressure from the first step** (constant, not ramped from 0), so that it acts
  before the minimum is chosen, possibly with a smaller grid ({0, ±1}, ±1/2).
- **Escapes that use the structure of the residual**: it is known explicitly (k rank-one
  tensors); e.g. add k terms that represent it (an exact 48 + k decomposition, close by) and
  reduce from there, or perturb the run along those k directions and restart (basin hopping).
- **Alternating least squares** with regularisation and rounding, the method that found most
  known schemes, as a reference for what a non-gradient method does from the same starts.

## Correction (27 October 2026): the learning-rate floor

The plateau scheduler used for these runs followed torch's `ReduceLROnPlateau`, which skips a
reduction smaller than 1e-8: halving from 1e-4 the learning rate stopped at 1.22·10⁻⁸, from 1e-3
at 1.53·10⁻⁸, whatever `--lr-min` below that. Consequences:

- the configurations "final lr 1e-10" (B-lr1e-4-min1e-10, B-lr1e-3-min1e-10) never went below
  1.2–1.5·10⁻⁸ and are the same runs as A-lr1e-4 and A-lr1e-3 (identical trajectories); the
  smallest final learning rate actually tried with the plateau schedule is ≈ 1.2·10⁻⁸ (and 1e-8
  with factor 0.1, stage C, and with the cosine schedule);
- with floor 1e-8 and factor 0.5 the stall rule, which waits for the floor, never fired: these
  runs took the whole budget (most of the 193 runs of stages A–C that ended on the budget).

The conclusions do not change (the minimum is chosen in the first 10⁴–10⁵ steps, where the
learning rate is still ≥ 1e-5). The scheduler is fixed in `matmulnet/schedule.py`: every
reduction is applied down to `--lr-min` (identical to torch above 2·10⁻⁸, hence for the
paper's protocol).

Rerun with the fixed scheduler (tags `B2-lr1e-4-min1e-10`, `B2-lr1e-3-min1e-10`, same starts,
40 runs): the learning rate now reaches 1e-10, every run stops on a stall (median 780,650 and
662,650 steps instead of the 2·10⁶ budget) and ends in the same minimum as the corresponding A
run (final ρ equal within 1e-3 in 40/40; 0 solved, median ρ 0.234 and 0.250). Rows in
`phase1b/rerun_B2.csv`.

## Reproduce

    python experiments/phase1.py starts --dir runs/round2-phase1 --scheme results/4x4-round-1/known_schemes/rational48.npz \
        --levels noise_s0.6 noise_s0.7 interp_t0.4 --seeds 1 2 3 4 5 6 7 8 9 10
    python experiments/phase1.py lm    --dir runs/round2-phase1 --cpus p --levels noise_s0.6 noise_s0.7 interp_t0.4 --seeds 1 2 3 4 5 6 7 8 9 10
    python experiments/phase1.py exact --dir runs/round2-phase1 --cpus p --levels noise_s0.7 interp_t0.4 --seeds 1 2 3 4 5 6 7 8 9 10 \
        --tag A-lr1e-3 -- --lr 1e-3 --lr-min 1e-8 --patience 10000
    python experiments/phase1.py exact ... --tag D2-q1e-3 -- --schedule cosine --lr 1e-3 --lr-min 1e-6 --quant 1e-3
    python experiments/phase1.py table --dir runs/round2-phase1 --levels noise_s0.7 interp_t0.4 --seeds 1 2 3 4 5 6 7 8 9 10 --methods lm "x:*"

(the options of every configuration: its tag in `phase1b/results.csv`, and its W&B config).
Cost: 550 runs, 39 CPU-hours, ≈ 6 hours of wall-clock time on 8 P-cores.
