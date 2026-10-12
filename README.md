# MatMulNet

Code for the paper *Neural Learning of Fast Matrix Multiplication Algorithms: a Geometric
Perspective* (P. Andreini, A. Bernardi, M. Bianchini, B. T. Corradini, S. Marziali,
G. Nunziati, F. Scarselli).

MatMulNet is a tensor network whose forward pass multiplies two `n x n` matrices with `r`
scalar multiplications,

    vec(AB)  ≈  F^T ( (H^T vec A) * (K^T vec B) ),

with learnable factors `H, K` of shape `(n^2, r)` and `F` of shape `(r, n^2)`. Training it on
random triples `(A, B, AB)` searches for a length-`r` decomposition of the matrix-multiplication
tensor `M_n`. In the border-rank variant the three factors are Laurent polynomials in a
parameter `ε` that decays during training.

The repository contains

| folder | content |
|---|---|
| `matmulnet/` | model, data, schedules and the training loop |
| `experiments/` | launcher of the training sweeps; known schemes for `4 x 4` matrices |
| `refinement/` | constrained refinement of a trained decomposition (projected Levenberg–Marquardt) |
| `continuation/` | projected pseudo-arclength continuation of the length-22 candidate, with its input and output data |
| `analysis/` | tables, statistical tests and figures |
| `tests/` | unit tests |

## Installation

Python 3.10 or later.

    pip install -r requirements.txt

Commands below are run from the root of the repository.

## Training

    python -m matmulnet.train --n 3 --rank 23 --seed 1 --out runs/rank_n3_r23_seed1
    python -m matmulnet.train --n 3 --rank 19 --degrees -2 4 --seed 1 --out runs/border_n3_r19_d-2_4_seed1

The protocol is the same for every experiment:

- training set of 10,000 and validation set of 1,000 triples `(A, B, AB)`, with the entries of
  `A` and `B` uniform in `[-1, 1]`; both sets are drawn once, and the training set is reshuffled
  at every epoch;
- all parameters initialised independently from `N(0, 1)` (`--init normal`, the default). For
  the border-rank model, `--init scaled` instead draws the coefficient of `ε^e` from `N(0, 1)`
  divided by `ε_0^e`, with `ε_0 = 1e-1` the initial value of `ε`, so that all monomials have
  unit size when training starts;
- loss: mean squared error per entry of the predicted product;
- Adam, mini-batches of 32, gradient norm clipped at 10;
- learning rate `1e-4`, halved when the validation loss has stopped improving for 15 epochs,
  down to `1e-6`;
- at most `10^6` epochs (`--max-epochs`);
- with `--degrees d_min d_max`, `ε` decays exponentially from `1e-1` to `1e-3` over the run;
- double precision throughout;
- the result of a run is its best validation loss, and the parameters that attain it.

A run writes, in its `--out` folder, `per_epoch.csv` (training and validation loss, learning
rate, `ε`, and the residual `ρ` and weight `W` of the current factors, every 100 epochs),
`best.npz` and `last.npz` (parameters and factors `H, K, F` at the best validation loss and at
the last epoch), `summary.json` (best and last validation loss with the corresponding `ρ` and
`W`, epochs, number of parameter updates, wall-clock time, arithmetic precision) and
`ckpt.npz`. The checkpoint holds the complete state — parameters, optimiser, schedules, random
generator, data — so a run that is interrupted and launched again with the same command
continues exactly as if it had never stopped.

`--init-from FILE` starts from the factors stored in `FILE` instead of a random point, and
`--wandb ENTITY/PROJECT` also sends the logged quantities to Weights & Biases (`pip install
wandb`).

### Two implementations of the training step

`matmulnet/reference_torch.py` is the PyTorch implementation: an `nn.Module`, autograd,
`clip_grad_norm_`, `torch.optim.Adam`. `matmulnet/kernels.py` computes the same epoch in a
single compiled function (numba), with the gradient written out by hand; it is the default
because it is more than an order of magnitude faster for models this small: about 0.5 ms per
epoch for `2 x 2`, 3–4.5 ms for `3 x 3` and 9–13 ms for `4 x 4` on one performance core of an
Intel Core i9-13900K (about twice as long on an efficiency core), i.e. up to four hours for a
`10^6`-epoch run. The kernel is compiled for the sizes of the run when the run starts (about two
seconds). Select the implementation with `--backend numba|torch`. The tests check
that the two produce the same trajectory up to rounding.

## The sweeps

    python experiments/sweep.py calibration    # 2x2, r = 6, 7                           (14 runs)
    python experiments/sweep.py rank           # 3x3, r = 16, ..., 24                    (63 runs)
    python experiments/sweep.py border         # 3x3, r = 16, ..., 21, exponent ranges
                                               # [-2,4], [-3,5], [-4,6]                  (126 runs)
    python experiments/sweep.py rank4          # 4x4, r = 45, ..., 49                    (35 runs)
    python experiments/sweep.py border4        # 4x4, r = 44, ..., 47, exponent range
                                               # [-1,2]                                  (28 runs)

Each configuration is run seven times, with seeds 1 to 7; `--ranks` and `--seeds` select a
subset or other values. Runs are independent single-threaded
processes, each pinned to its own physical core: `--cpus` chooses the cores (`all`, the default,
one per physical core; `p` or `e`, only the performance or the efficiency cores of a hybrid Intel
CPU; or a list such as `0,2,4-7`), performance cores first, and `--jobs` runs fewer at once (for
instance on a shared machine). Finished runs are skipped and interrupted ones
resume, so the same command can be issued again at any time. `border --init scaled` runs the
border-rank sweep with the scaled initialisation; its runs are stored next to the others (with
`_scaled` in the name), and `analysis/stats.py border` and `analysis/figures.py border` take the
same `--init` option to select them.

## Analysis

    python analysis/collect.py                 # runs/*/summary.json  ->  results.csv
    python analysis/stats.py calibration       # 2x2: summary and successes below 1e-12
    python analysis/stats.py rank              # 3x3: summary, residual ρ, Fisher's exact test,
                                               # Jonckheere–Terpstra, Welch tests with Holm
                                               # correction and Cohen's d
    python analysis/stats.py border            # per exponent range: summary and
                                               # Jonckheere–Terpstra; two-way ANOVA; Welch tests
    python analysis/stats.py precision         # dynamic range of the monomials in ε
    python analysis/figures.py rank            # figures/rank_strip_{linear,log}
    python analysis/figures.py border          # figures/brank_strip

## Limit of an ε-parametrised decomposition

    python analysis/laurent.py runs/border_n4_r46_d-1_2_seed1/best.npz

With the coefficients held fixed, the tensor represented by the border-rank model is a Laurent
polynomial `T(ε) = Σ_d ε^d T_d`, and the family converges to `M_n` as `ε → 0` exactly when
`T_d = 0` for all `d < 0` and `T_0 = M_n`. The script reports the norm of every `T_d`, the
distance of `T_0` from `M_n`, and `ρ`, `W` and the cancellation level of the factors evaluated
on a decreasing sequence of values of `ε`.

## Known schemes for 4 x 4 matrices

    python experiments/known_schemes.py strassen49 --out schemes/strassen49.npz
    python experiments/known_schemes.py rational48 --source 4x4x4_LRP.mpl --out schemes/rational48.npz
    python experiments/known_schemes.py smirnov46  --source smirnov.txt --out schemes/smirnov46.npz
    python experiments/known_schemes.py perturb schemes/rational48.npz --sigma 1e-2 --out schemes/rational48_s1e-2.npz

write, in the format of `best.npz`, Strassen's scheme applied to itself (49 terms), the
rational 48-term scheme of Dumas, Pernet and Sedoglavic (arXiv:2506.13242; `--source` is the
file of the catalogue <https://fmm.univ-lille.fr/4x4x4.html>), the approximate scheme of length
46 of Smirnov (arXiv:1412.1687; `--source` is the output of `pdftotext -layout` on the paper),
whose coefficients are Laurent polynomials in `ε` with exponents `-1, …, 2`, and a noisy copy
of a stored scheme. Each scheme is checked against `M_4` when it is written. The files can be
passed to `matmulnet.train --init-from`, `refinement.refine --init` and `analysis/laurent.py`.

## Constrained refinement

    python -m refinement.refine --init runs/rank_n3_r22_seed<i>/best.npz --out refinement_r22
    python -m refinement.refine --init runs/rank_n3_r23_seed<j>/best.npz --out refinement_r23
    python analysis/figures.py frontier        # figures/frontier_r22_r23

where `<i>` and `<j>` are the best runs at `r = 22` and `r = 23` (lowest `best_val_loss` in
`results.csv`). For an increasing sequence of bounds `c` (0.1, 0.2, …, 0.9, 1, 2, …, 25) the
script minimises the full-tensor relative residual

    ρ(θ) = ||T_θ − M_n||_F / ||M_n||_F      subject to   |θ_i| ≤ c

by projected Levenberg–Marquardt, each level being warm-started from the previous one. It
records `ρ` against the gauge-invariant weight `W = Σ_s ||h_s|| ||k_s|| ||f_s||` in
`frontier.csv`, saves the factors of every level, and stops when `ρ` falls below the
double-precision cancellation level of the current solution. `--max-seconds` bounds the time
spent on each level and `--threads` sets the threads of the linear algebra (useful from
`4 x 4` on, where an iteration solves a system with a few thousand unknowns: there 4 threads
are about three times as fast as one, and more threads gain little). The method is local: the values
are those reached along the warm-started branch, not global minima over the boxes.

## Pseudo-arclength continuation

    cd continuation
    python continuation.py --self-test
    python continuation.py rank22_input.csv --out replay
    python verify_saved_points.py replay

`rank22_input.csv` holds the coefficients of the length-22 candidate from which the
continuation starts. `continuation/data/` is the output of the same command with `--out data`,
the run described in the paper: the
projection `P`, the slice matrices and the initial singular values (`system.npz`), the
diagnostics of all accepted steps (`trace.csv`), the factors at the smallest recorded `t`, at
the smallest full-tensor residual and at the last accepted point (`best_t.npz`, `best_rho.npz`,
`last.npz`), the options and software versions (`metadata.json`, `summary.json`) and the
extended-precision checks of the saved points (`verification.json`). The script refuses to
overwrite an existing output folder.

## Tests

    python -m pytest tests

## Conventions

Matrices are flattened row-major: entry `(i, j)` has index `n·i + j`. Summand `s` of a
decomposition is `(h_s, k_s, f_s) = (H[:, s], K[:, s], F[s, :])`, the represented tensor is
`T[a, b, c] = Σ_s H[a, s] K[b, s] F[s, c]`, and `M_n[n·i+j, n·j+k, n·i+k] = 1`. In the
border-rank model the parameters are coefficient tensors whose last axis is indexed by the
exponent, `H(ε) = Σ_e ε^e H_coeff[:, :, e − d_min]`, and likewise for `K` and `F`.

## Licence

MIT, see `LICENSE`.
