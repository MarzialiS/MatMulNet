# 4×4 round 2 — phase 1: recovering a known 48-term decomposition

Run 13–14 October 2026 on the i9-13900K (8 P-cores, one single-threaded process per core).
Code: branch `4x4-round-2` (`experiments/phase1.py`, options of `matmulnet/train.py`).
Raw runs: `runs/round2-phase1/` on the machine (not in git). Every run is also on W&B,
project `sara-marziali-university-of-siena/matmulnet-4x4`, tags `round-2`, `phase-1`,
job types `phase1-lm-only`, `phase1-train`, `phase1-reduce`.

## Question

Starting from the rational 48-term scheme of Dumas–Pernet–Sedoglavic, perturbed more and more,
does training bring it back to an exact 48-term decomposition, and with which hyperparameters?
A training method only counts where the perturbation is strong enough that Levenberg–Marquardt
alone does not already solve it.

## Set-up

- **Starts** (seeds 1–5 each): `noise` — the scheme plus Gaussian noise, σ ∈ {0.1, 0.2, 0.3,
  0.5, 0.7, 1} times the r.m.s. of its non-zero coefficients (0.845); `interp` — (1 − t) · scheme
  + t · random start drawn as the training runs draw it (N(0, 1)), t ∈ {0.1, …, 0.9}. All are far
  from the scheme in residual: ρ = 1.2 already at σ = 0.1, ρ ≈ 11 at t = 0.5.
- **Solved**: ρ < 1e-10 with W < 1e4 at the end (W of the scheme: 210.5).
- **Control** (`lm`): unconstrained LM from the start, 1000 iterations (≈ 3.5 min).
- **Training**: 50,000 epochs from the start, then at most 200 iterations of the same LM
  (`--polish 200`); a training run is solved if the polished point is.

## Results (solved / 5 starts per level)

| level | LM alone | protocol | 15 other configurations |
|---|---|---|---|
| σ = 0.1, 0.2, 0.3, 0.5 | 5/5 each | 5/5 each | not run (control already solves) |
| σ = 0.7 | 0/5 | **1/5** | not run |
| σ = 1 | 0/5 | 0/5 | 0/5 each |
| t = 0.1, 0.2 | 5/5 each | 5/5 each | not run |
| t = 0.3 | 1/5 | **5/5** | not run |
| t = 0.4, 0.6, 0.8 | 0/5 each | 0/5 each | 0/5 each |
| t = 0.5, 0.7, 0.9 | 0/5 each | 0/5 each | not run |

The 15 other configurations, each on σ = 1 and t = 0.4, 0.6, 0.8 (20 runs each, 300 runs):

| family | configurations |
|---|---|
| learning rate and schedule | constant 1e-4, constant 1e-3; cosine from 1e-3, 3e-3, 1e-2; plateau from 1e-3 with patience 1000; cosine 1e-3 with batch 128 |
| AdamW weight decay (cosine 1e-3) | 1e-3, 1e-2, 1e-1 |
| over-parametrisation, 52 terms + group lasso on the terms, pruned to 48 (cosine 1e-3) | λ = 1e-2, 1e-1, 1 |
| rank reduction 52 → 48 from the λ = 1e-2 runs: drop the least useful term, train 10,000 epochs, repeat | one |

None solves a single start beyond the protocol. Full table: `phase1/table.txt`; every run:
`phase1/results.csv`.

## What the runs show

1. **Where training helps, it is the training itself**: at t = 0.3 the protocol reaches
   ρ ≈ 1e-7 (validation loss ≈ 4e-15) by itself in ≈ 300 epochs; the polish only finishes it.
   At σ = 0.7 the one success ends training at ρ = 0.17, W = 378, and the polish takes it to 2e-15
   — training moved it into the right basin, LM alone from the start did not get there.
2. **Where it fails, it fails early and the same way for every configuration**: ρ settles within
   5,000–10,000 epochs at 0.18–0.29 and then stays flat; longer runs would not help. The weight
   at the end of training is moderate (median 380–1600 depending on the configuration, without
   weight decay), so these
   are spurious local minima, not border-rank degenerations. The residual is spread over many
   entries (the largest single entry carries at most ≈ 9% of ‖T − M‖²).
3. **The large W reported by the polish is produced by the polish**: when LM fails from these
   minima it drifts towards ρ ≈ √(2/64), √(3/64), √(4/64) = 0.177, 0.217, 0.250 ("M₄ minus 2, 3, 4
   entries", the basins seen in round 1) with W in the thousands.
4. **Learning rate, schedule, batch** (1e-4 to 1e-2, constant / cosine / plateau with long
   patience, batch 128): no effect on the outcome. The protocol's plateau schedule, which drops
   the learning rate to 1e-6 within a few thousand epochs, is not what blocks phase 1.
5. **Weight decay** pulls W down (median 283 at 1e-2, 119 at 1e-1 — below the scheme's 210) but
   ρ does not move (0.22, 0.29).
6. **52 terms** reach ρ ≈ 0.01–0.02 (group lasso 1e-2), far below the 48-term minima — the
   extra terms do open a way down. But the group lasso never selects terms: it shrinks all of
   them alike (smallest term weights ≈ 1 at every λ), and every term carries real work — even the
   least useful one raises ρ from ≈ 0.015 to ≈ 0.18 when removed. Pruning, at once or one term at
   a time with retraining, lands back in the 48-term minima (ρ ≈ 0.18–0.25).
7. The solutions found are never the scheme's own terms (distance of the rank-one terms ≈ 0.1
   even at σ = 0.1): they are other points of the same family of decompositions, moved along its
   symmetries. Every success is "another decomposition".

## Conclusion for phase 2

- The usable algorithm is the protocol's training followed by a short LM polish. It extends the
  reach of LM alone from t = 0.2 to t = 0.3, and from σ = 0.5 to σ = 0.7 in one case out of five.
  No variant tried extends it further.
- The obstacle is the landscape at 48 terms (spurious minima at ρ ≈ 0.2 reached early), not the
  optimiser's settings. Phase 2 (48 from scratch, i.e. t = 1) is therefore expected to fail with
  every configuration tried here.
- Directions not tried yet, if phase 2 is to have a chance: many cheap restarts (the plateau is
  reached by 5,000–10,000 epochs, ≈ 1 minute per attempt) with the success rate measured as a
  function of the budget; a pressure towards the discrete structure of known schemes
  (coefficients in {0, ±1, ±1/2, …}, sparsity); starting from the 49-term scheme and removing a
  term, as a different reduction path; training on the full-tensor loss instead of sampled
  triples.

## Reproduce

    python experiments/phase1.py starts --dir runs/round2-phase1 --scheme results/4x4-round-1/known_schemes/rational48.npz
    python experiments/phase1.py lm     --dir runs/round2-phase1 --cpus p --wandb ENTITY/PROJECT
    python experiments/phase1.py train  --dir runs/round2-phase1 --cpus p --tag protocol --max-epochs 50000
    python experiments/phase1.py train  --dir runs/round2-phase1 --cpus p --levels noise_s1 interp_t0.4 interp_t0.6 interp_t0.8 \
        --tag cos1e-3 --max-epochs 50000 -- --schedule cosine --lr 1e-3
    python experiments/phase1.py reduce --dir runs/round2-phase1 --cpus p --levels noise_s1 interp_t0.4 interp_t0.6 interp_t0.8 \
        --tag from_r52gl1e-2 --source r52gl1e-2 --stage-epochs 10000
    python experiments/phase1.py table  --dir runs/round2-phase1

(the other configurations: tags and options in `phase1/results.csv` and on W&B). Cost of the
phase: ≈ 8.5 hours of wall-clock time on 8 P-cores, 9.6–10.5 ms per epoch at r = 48.
