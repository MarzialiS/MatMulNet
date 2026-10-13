# 4×4 round 1 — handoff

Experiments on the 4×4 matrix-multiplication tensor M₄, run 9–12 October 2026 on one laptop
(Intel Core Ultra 9 285H, P-cores 0–5, one single-threaded process per run). This file is the
journal of the round: what was asked, what was decided, what was run, what came out, and what
is still open. Everything it refers to is in this folder or on the `main` branch of the code.

**Status of this branch.** The repository is private. This branch holds results and process
notes that are not meant for the public version of the code: before the repository is made
public, this branch (and in particular `known_schemes/`, which holds third-party coefficients)
has to be removed or moved elsewhere.

## 1. Why this round

A reader of the paper suggested that it would be much stronger with a new case, naming 4×4
matrix multiplication as wide open. The co-authors asked for a first round on M₄ with the
protocol of the 3×3 experiments. Known bounds used as reference:

- rank: 34 ≤ R(M₄) ≤ 48 (lower bound Bläser; upper bound AlphaEvolve, rational version by
  Dumas–Pernet–Sedoglavic);
- border rank: 29 ≤ R̲(M₄) ≤ 46 (lower bound Landsberg–Michałek; upper bound Smirnov).

The request, in short: (1) rank sweep r = 45…49, starting from 47, 48, 49, then seven
independent runs per rank, recording updates, epochs and time; controls with the known 49-term
and 48-term schemes, also slightly perturbed; (2) for every run save H, K, F at the best
checkpoint and at the end, follow the full-tensor relative residual and the weight
W = Σ ‖h‖‖k‖‖f‖, and repeat the constrained refinement on the best candidates at 47 and 48;
(3) border rank at r = 44…47, after checking that the model can represent Smirnov's 46-term
family and choosing the exponent range from it; for each candidate, evaluate the same family
at fixed coefficients on decreasing ε and check the Laurent coefficients of the total tensor
(negative degrees must cancel, the constant coefficient must be M₄).

## 2. Decisions taken before launching

| question | decision |
|---|---|
| ranks if the pilot at 47, 48, 49 shows no success | extend upwards with one seed per rank |
| border-rank protocol | check Smirnov's family first, then the paper's protocol with `--init scaled` |
| machine | the laptop, six P-cores, launched on request and left running unattended |
| where the code lives | in this repository, not published |
| experiment tracking | Weights & Biases, project `sara-marziali-university-of-siena/matmulnet-4x4`, one project; `job_type` rank / border / control, one group per configuration |
| known schemes | fetched from the authors' published sources and accepted only if they rebuild M₄ |
| refinement budget | the paper's chain of bounds, with a time limit per level |

Choices made without asking, all reversible: protocol identical to 3×3 (10⁶ epochs, seeds 1–7);
ρ and W logged every 100 epochs; `last.npz` written next to `best.npz`; perturbation sizes
10⁻³, 10⁻², 10⁻¹ (relative to the r.m.s. of the non-zero coefficients).

## 3. Code written for the round (all on `main`)

- `matmulnet/train.py`: logs ρ and W, writes `last.npz`, records updates, wall-clock time and
  precision in `summary.json`, `--init-from FILE`, `--wandb ENTITY/PROJECT`.
- `experiments/sweep.py`: experiments `rank4` (r = 45…49) and `border4` (r = 44…47, exponents
  [−1, 2]); `--ranks`, `--seeds`, `--wandb`.
- `experiments/known_schemes.py`: Strassen ⊗ Strassen (49 terms), converter for the rational
  48-term scheme (file `4x4x4_LRP.mpl` of https://fmm.univ-lille.fr/4x4x4.html), converter for
  Smirnov's 46-term family (text of arXiv:1412.1687 via `pdftotext -layout`; the paper lists
  the coefficients evaluated at ε = 0.1, and all of them are monomials ±ε^k except one
  binomial), and `perturb`.
- `analysis/laurent.py`: Laurent coefficients T_d of the tensor at fixed coefficients,
  ‖T₀ − M‖/‖M‖, and ρ, W, cancellation level on a decreasing sequence of ε.
- `refinement/refine.py`: from 4×4 on, the normal equations are assembled from Gram matrices
  without forming the Jacobian (for n ≤ 3 the original path is kept and gives the same chain
  as before, checked on the 3×3 r = 23 chain: machine precision at c = 0.8, W = 65.4);
  `--max-seconds` per level; `--threads`.

Index conventions (unchanged): row-major flattening, T[a,b,c] = Σ_s H[a,s] K[b,s] F[s,c],
M[4i+j, 4j+k, 4i+k] = 1. For 4×4 with entries uniform in [−1, 1] the expected loss is
**4ρ²/9** (it is ρ²·n/9; the 3×3 identity ρ²/3 is the case n = 3).

## 4. Journal

**9 Oct, evening — preparation and controls.**
- W&B: write access to the team needed a new API key; project created.
- Known schemes found and verified: 49 (ρ = 0), 48 rational (ρ = 0), Smirnov 46 (negative
  degrees exactly zero, constant term exactly M₄, ρ ≈ 0.76·ε). Smirnov's exponents are
  −1…2, hence the range [−1, 2] for the border-rank model.
- Timing on one core: 9 ms/epoch for rank r = 48, 14 ms/epoch for border r = 46.
- Training from the exact 48- and 49-term schemes perturbed by 10⁻³, 10⁻², 10⁻¹, 20,000
  epochs: best loss 3×10⁻¹⁴ … 7×10⁻¹¹ in all six runs; the loss at the last epoch is about
  10⁻¹⁰ (Adam at the learning-rate floor does not sit still on a solution).
- Refinement at a single level c = 2 from the schemes perturbed by 1%: ρ ≈ 7×10⁻¹⁶ in 6–11
  iterations.
- **Smirnov's family is not held by training.** Starting exactly on it, validation loss
  after one epoch at fixed ε:

  | ε | loss of the family | lr 10⁻⁴ | lr 10⁻⁶ | lr 10⁻⁸ |
  |---|---|---|---|---|
  | 10⁻¹ | 2.7×10⁻³ | 0.39 | 1.8×10⁻³ | 2.6×10⁻³ |
  | 10⁻² | 2.6×10⁻⁵ | 3×10⁵ | 33 | 2.6×10⁻³ |
  | 10⁻³ | 2.6×10⁻⁷ | 3×10¹¹ | 3×10⁷ | 3×10³ |

- **The paper's chain (c from 0.1 to 25) started from the known schemes perturbed by 1% does
  not come back to them.** The first level clips every coefficient to 0.1 and the chain then
  ends at ρ = 0.2165 = √(3/64) (49 terms) and ρ = 0.2500 = √(4/64) (48 terms), with W growing
  to 5×10⁴. The same happens in 3×3: the chain started from the best of r = 23, seed 1
  (ρ = 0.011) ends at ρ = 0.1925, with the old and the new code alike; the chain that reaches
  machine precision at c = 0.8 starts from a different point.

**10 Oct, 00:56 — rank, first block.** r = 47, 48, 49, 50, 52, 56, seed 1, six P-cores,
about 12 ms/epoch with six runs in parallel (3.3 h per run). No run below 10⁻³ at 47…50 and
56; r = 52 at 9.7×10⁻⁴. In every run the best is in the last epochs, and the learning rate
reaches its floor 10⁻⁶ within the first 10,000 epochs.

**10 Oct, 04:22 – 11 Oct, 00:42 — rest of the rank sweep.** r = 45, 46, 51, 54, 60, 64 with
seed 1, then seeds 2–7 for r = 45…49. In parallel, on the E-cores, direct refinement of the
r = 52 and r = 49 seed-1 candidates (20 minutes per level).

**11 Oct, 00:42 — border sweep and refinements.** `border4 --init scaled` on the P-cores; the
control run started from Smirnov's family on one E-core; direct refinement (one hour per
level, bounds starting at the smallest integer that does not clip the candidate) of the best
candidate at r = 48, 47, 49 and of r = 54, 60, 51 on six E-cores. The first border block ran
at 26 ms/epoch because of the refinements; afterwards 17.5 ms/epoch (4.9 h per run).

**11 Oct, 16:52 – 19:31 — pause.** All runs stopped on request and resumed from their
checkpoints (the six interrupted runs were at 82–92%).

**12 Oct, 06:20 — border sweep finished.** `analysis/laurent.py` run on every `best.npz` and
`last.npz`. Tables and figures built and published as a page (see section 7).

## 5. Results

### Rank (10⁶ epochs, validation loss at the best checkpoint)

| r | runs | mean | min | max | median ρ | min ρ | below 10⁻³ |
|---|---|---|---|---|---|---|---|
| 45 | 7 | 5.5×10⁻² | 4.3×10⁻² | 6.4×10⁻² | 0.359 | 0.313 | 0 |
| 46 | 7 | 4.6×10⁻² | 3.6×10⁻² | 5.2×10⁻² | 0.336 | 0.284 | 0 |
| 47 | 7 | 3.5×10⁻² | 3.0×10⁻² | 4.3×10⁻² | 0.270 | 0.255 | 0 |
| 48 | 7 | 3.3×10⁻² | 2.6×10⁻² | 3.7×10⁻² | 0.279 | 0.255 | 0 |
| 49 | 7 | 2.6×10⁻² | 2.0×10⁻² | 3.6×10⁻² | 0.240 | 0.222 | 0 |

One seed per rank above: 50: 2.2×10⁻²; 51: 1.3×10⁻³; 52: 9.7×10⁻⁴; 54: 4.6×10⁻⁵;
56: 8.7×10⁻³; 60: 1.2×10⁻⁴; 64: 2.7×10⁻³.

- No transition at 48 or 49: the mean decreases gradually, and 47 vs 48 is within the spread
  of the seeds.
- Above 50 the loss drops by orders of magnitude but not monotonically; at r = 64 training
  does not find the standard algorithm.
- No run had converged: the best is always in the last epochs.

### Border rank (exponents [−1, 2], scaled initialisation, 10⁶ epochs)

| r | mean | min | max | median ε at best | median negative part | median ‖T₀ − M₄‖/‖M₄‖ | median loss at the end |
|---|---|---|---|---|---|---|---|
| 44 | 6.2×10⁻² | 5.6×10⁻² | 7.9×10⁻² | 0.039 | 8.5 | 119 | 2.8×10⁴ |
| 45 | 5.5×10⁻² | 4.8×10⁻² | 6.5×10⁻² | 0.046 | 10.0 | 133 | 5.5×10⁴ |
| 46 | 5.0×10⁻² | 3.6×10⁻² | 6.5×10⁻² | 0.041 | 9.3 | 129 | 4.4×10⁴ |
| 47 | 3.7×10⁻² | 2.9×10⁻² | 4.7×10⁻² | 0.043 | 10.0 | 132 | 5.6×10⁴ |

- The limit check fails on all 28 bests: the negative degrees do not cancel and the constant
  term is more than a hundred norms away from M₄. At fixed coefficients ρ ≈ 10⁴ at ε = 10⁻²
  and ≈ 10⁷ at ε = 10⁻³. The bests are fits at a fixed ε ≈ 0.04, at the same loss level as
  the rank model with the same r.
- The loss reaches its minimum early (ε between 0.024 and 0.084), then rises and blows up as
  ε goes to 10⁻³.
- Control from Smirnov's family (r = 46, full schedule): best 4.8×10⁻⁴ at ε = 0.079, then the
  family is lost; 3×10⁷ at the end.

### Refinement of the rank candidates (direct, no descent to small bounds)

| r (seed) | start ρ | first level | c = 10 | c = 25 | final W |
|---|---|---|---|---|---|
| 47 (1) | 0.255 | 0.2509 | 0.2501 | 0.2500 | 1.7×10⁵ |
| 48 (4) | 0.255 | 0.2502 | 0.2500 | 0.2500 | 5.7×10⁴ |
| 49 (6) | 0.222 | 0.1796 | 0.1770 | 0.1768 | 1.5×10⁵ |
| 49 (1) | 0.253 | 0.2171 | 0.2166 | 0.1768 | 1.3×10⁵ |
| 51 (1) | 0.053 | 2.5×10⁻² | 3.0×10⁻³ | 3.4×10⁻⁴ | 1.8×10⁵ |
| 52 (1) | 0.047 | 1.7×10⁻² | 4.7×10⁻³ | 7.3×10⁻⁴ | 2.2×10⁵ |
| 54 (1) | 0.010 | 2.4×10⁻³ | 1.7×10⁻⁴ | 1.5×10⁻⁴ | 5.7×10⁴ |
| 60 (1) | 0.016 | 3.7×10⁻⁴ | 2.4×10⁻⁶ | 1.9×10⁻⁶ | 6.4×10³ |

- At 47, 48, 49 the residual stops on √(4/64) = 0.2500 and √(2/64) = 0.1768: the refinement
  converges to "M₄ minus 4 entries" and "M₄ minus 2 entries" while the weight grows.
- r = 60 is the only one approaching a solution at bounded weight; it stopped on the time
  limit, not at machine precision.
- At 51, 52, 54 the residual falls only while the weight grows (ρ·W roughly constant at
  r = 52: 125 → 163). This is the signature read in the paper as a border-rank degeneration
  at r = 22, and it appears here above the known rank.
- Almost every level stopped on the time limit: these numbers do not exclude that with more
  time the residual collapses at bounded weight.

## 6. What this means, and what is open

1. With the 3×3 protocol, 4×4 does not reproduce the 22/23 behaviour: the loss decreases with
   r, there is no threshold at 48/49.
2. The bottleneck is the optimisation, not the rank: r = 64 fails, the learning-rate schedule
   collapses to its floor within 10,000 epochs, and the controls from known schemes work. The
   first thing to try in a second round is a learning-rate schedule that does not collapse
   (to be declared as a change of protocol), and more seeds above 50.
3. Border rank, checked as a limit, gives no degeneration, and training cannot hold even
   Smirnov's. This also bears on how the border-rank section of the paper is read. Possible
   directions: a parametrisation in which all monomials have comparable gradients, per-degree
   learning rates, or a second-order method on the border model.
4. Two warnings on the paper's refinement: the chain depends on the branch (it fails from
   exact schemes too), and "ρ falls while W grows" also appears where exact decompositions
   exist. A direct refinement (bounds starting above the candidate's largest coefficient) is
   proposed as the main tool for 4×4, with the paper's chain as a comparison.
5. To verify: the failure basin of the 3×3 runs (ρ ≈ 0.194–0.197) is close to
   √(1/27) = 0.1925, i.e. "M₃ minus one entry". Not checked on the paper's data.

None of points 1–5 has been discussed with the co-authors yet.

## 7. What is in this folder

- `training/<run>/`: `summary.json`, `best.npz`, `last.npz` (coefficients, factors, ε, epoch,
  loss), `per_epoch.csv.gz` (every 100 epochs: ε, learning rate, training and validation
  loss, ρ, W) and, for the border runs, `best.laurent.json`, `last.laurent.json`. 35 + 7 rank
  runs, 28 border runs, 7 control runs. The checkpoints (`ckpt.npz`, 322 MB) are not
  included; they are on the laptop in `runs/` and allow an exact restart.
- `refinement/`: `direct_rank_n4_r*_seed*` (candidates), `chain_*_s1e-2` and `direct_*_s1e-2`
  (controls from the known schemes); each with `frontier.csv` and the factors of every level.
- `known_schemes/`: the three schemes and the perturbed copies used as starting points.
- `report/`: `gather.py` → `data.json`, `build.py` + `style.css` → `matmulnet-4x4.html`
  (11 tables, 6 figures, in Italian). Published at
  https://claude.ai/artifact/WRYqF6saKk3M1Fk6HFh2gD (private; shared from the page's menu).
  Note: its timing table gives 19.4 ms/epoch for the border runs, an average that includes
  the slowed first block; on a dedicated machine it is about 17.5 ms.

## 8. How to reproduce or continue

From the root of the repository, with the environment of `requirements.txt`:

    # known schemes (sources: fmm.univ-lille.fr/4x4x4_LRP.mpl.bz2, arXiv:1412.1687)
    python experiments/known_schemes.py strassen49 --out schemes/strassen49.npz
    python experiments/known_schemes.py rational48 --source 4x4x4_LRP.mpl --out schemes/rational48.npz
    python experiments/known_schemes.py smirnov46  --source smirnov.txt   --out schemes/smirnov46.npz

    # sweeps (finished runs are skipped, interrupted ones resume)
    taskset -c 0-5 python experiments/sweep.py rank4 --jobs 6 --wandb ENTITY/PROJECT
    taskset -c 0-5 python experiments/sweep.py rank4 --ranks 50 51 52 54 56 60 64 --seeds 1 --jobs 6
    taskset -c 0-5 python experiments/sweep.py border4 --init scaled --jobs 6

    # control from Smirnov's family
    python -m matmulnet.train --n 4 --rank 46 --degrees -1 2 --seed 1 \
        --init-from schemes/smirnov46.npz --out runs/control_smirnov46_train

    # limit check and refinement
    python analysis/laurent.py runs/border_n4_*/best.npz
    python -m refinement.refine --init runs/rank_n4_r48_seed4/best.npz --out refinement_n4/direct_rank_n4_r48_seed4 \
        --levels 4 10 25 --max-seconds 3600 --threads 2

    # tables and figures
    python results/4x4-round-1/report/gather.py && python results/4x4-round-1/report/build.py

Practical notes: seeds are those of `numpy.random.default_rng`; a run draws its data, then
its initial point, from the same generator. Running refinements on the E-cores while the
sweep runs on the P-cores slows the sweep by about 50%. Stop everything with
`pkill -f experiments/sweep.py; pkill -f matmulnet.train`; the same sweep command resumes.
