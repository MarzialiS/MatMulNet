"""Known schemes for 4 x 4 matrix multiplication, in the format of the training runs.

    python experiments/known_schemes.py strassen49 --out schemes/strassen49.npz
    python experiments/known_schemes.py rational48 --source 4x4x4_LRP.mpl --out schemes/rational48.npz
    python experiments/known_schemes.py smirnov46 --source smirnov.txt --out schemes/smirnov46.npz
    python experiments/known_schemes.py perturb schemes/rational48.npz --sigma 1e-2 --seed 1 \\
        --out schemes/rational48_s1e-2.npz

strassen49   Strassen's 7-term scheme for 2 x 2 matrices applied to itself (49 terms).
rational48   The 48-term scheme with rational coefficients of Dumas, Pernet and Sedoglavic
             (arXiv:2506.13242). --source is the file 4x4x4_LRP.mpl of the catalogue
             https://fmm.univ-lille.fr/4x4x4.html (bunzip2 the download).
smirnov46    The approximate scheme of length 46 of Smirnov (arXiv:1412.1687), a family of
             46-term decompositions with Laurent-polynomial coefficients in eps, exponents
             -1, ..., 2, converging to M_4 as eps -> 0. --source is the text of the paper
             (`pdftotext -layout`), which lists the coefficients evaluated at eps = 0.1.
perturb      Add Gaussian noise of standard deviation sigma times the root mean square of the
             non-zero coefficients to every coefficient of a stored scheme.

Every scheme is checked against M_4 before it is written. The output holds H_coeff, K_coeff,
F_coeff and degrees, and for the exact schemes also the factors H, K, F; it can be passed to
`matmulnet.train --init-from`, `refinement.refine --init` and `analysis/laurent.py`.
"""
import argparse
import itertools
import os
import re
import sys
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from matmulnet import metrics  # noqa: E402


def strassen():
    """Strassen's scheme for 2 x 2 matrices: H, K of shape (4, 7), F of shape (7, 4)."""
    H = np.array([[1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
                  [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]], dtype=float).T
    K = np.array([[1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
                  [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]], dtype=float).T
    F = np.array([[1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1], [1, 0, 1, 0],
                  [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]], dtype=float)
    return H, K, F


def kronecker(a, b, na, nb):
    """Scheme for (na nb) x (na nb) matrices from schemes for na x na and nb x nb matrices:
    the first acts on the blocks, the second inside them."""
    n = na * nb
    # position of block entry (I, J), inner entry (i, j) in the row-major n x n matrix
    pos = np.array([[(nb * I + i) * n + nb * J + j for i in range(nb) for j in range(nb)]
                    for I in range(na) for J in range(na)]).ravel()
    Ha, Ka, Fa = a
    Hb, Kb, Fb = b
    H, K, F = np.kron(Ha, Hb), np.kron(Ka, Kb), np.kron(Fa, Fb)
    out_h, out_k, out_f = np.zeros_like(H), np.zeros_like(K), np.zeros_like(F)
    out_h[pos], out_k[pos], out_f[:, pos] = H, K, F
    return out_h, out_k, out_f


def rational48(source):
    text = open(source).read()
    mats = []
    for found in re.finditer(r"Matrix\((\d+), *(\d+), *\[(.*?)\]\]\)", text, re.S):
        rows, cols = int(found.group(1)), int(found.group(2))
        vals = [float(Fraction(t)) for t in re.findall(r"-?\d+(?:/\d+)?", found.group(3))]
        mats.append(np.array(vals).reshape(rows, cols))
    L, R, P = mats
    t = np.arange(16).reshape(4, 4).T.ravel()        # transposition of a flattened matrix
    return L[:, t].T.copy(), R[:, t].T.copy(), P.T.copy()


def smirnov46(source):
    """Returns coefficient tensors (H_coeff, K_coeff, F_coeff) and the exponents."""
    degrees = [-1, 0, 1, 2]
    blocks, cur = [], None
    for line in open(source).read().split("\n"):
        if re.match(r"\s*t = \d+\s*$", line):
            cur = []
            blocks.append(cur)
            continue
        tok = line.split()
        if (cur is not None and len(cur) < 4 and len(tok) == 12
                and all(re.fullmatch(r"-?\d+(\.\d+)?", x) for x in tok)):
            cur.append([float(x) for x in tok])
    values = np.array(blocks)                         # (46, 4, 12): three 4 x 4 matrices per term

    def decode(v):
        """The polynomial with coefficients in {-1, 0, 1} and fewest terms whose value at 0.1 is v."""
        best = None
        for c in itertools.product((-1, 0, 1), repeat=len(degrees)):
            if abs(sum(s * 0.1 ** d for s, d in zip(c, degrees)) - v) < 1e-9:
                if best is None or sum(map(abs, c)) < sum(map(abs, best)):
                    best = c
        if best is None:
            raise ValueError(f"cannot decode {v}")
        return best

    coeff = np.array([decode(v) for v in values.ravel()], dtype=float).reshape(46, 4, 12, 4)
    a, b, c = (coeff[:, :, 4 * i:4 * i + 4, :] for i in range(3))
    flat = lambda x: x.reshape(46, 16, 4)             # noqa: E731
    Hc = flat(a.transpose(0, 2, 1, 3)).transpose(1, 0, 2)
    Kc = flat(b).transpose(1, 0, 2)
    Fc = flat(c.transpose(0, 2, 1, 3))
    return (Hc.copy(), Kc.copy(), Fc.copy()), degrees


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scheme", choices=["strassen49", "rational48", "smirnov46", "perturb"])
    ap.add_argument("input", nargs="?", help="perturb: the stored scheme")
    ap.add_argument("--source", help="file with the published coefficients")
    ap.add_argument("--sigma", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    if a.scheme == "perturb":
        z = np.load(a.input)
        rng = np.random.default_rng(a.seed)
        deg = z["degrees"]
        coeff = []
        for k in ("H_coeff", "K_coeff", "F_coeff"):
            c = z[k]
            coeff.append(c + a.sigma * np.sqrt(np.mean(c[c != 0] ** 2)) * rng.standard_normal(c.shape))
    elif a.scheme == "smirnov46":
        coeff, deg = smirnov46(a.source)
        lau = metrics.laurent(*coeff, deg)
        target = metrics.matmul_tensor(4)
        assert all(np.all(t == 0) for d, t in lau.items() if d < 0) and np.all(lau[0] == target)
    else:
        H, K, F = kronecker(strassen(), strassen(), 2, 2) if a.scheme == "strassen49" \
            else rational48(a.source)
        assert metrics.rho(H, K, F, 4) < 1e-14
        coeff, deg = [H[..., None], K[..., None], F[..., None]], [0]

    extra = {}
    if list(deg) == [0]:
        extra = dict(H=coeff[0][..., 0], K=coeff[1][..., 0], F=coeff[2][..., 0])
    np.savez(a.out, H_coeff=coeff[0], K_coeff=coeff[1], F_coeff=coeff[2], degrees=np.asarray(deg),
             n=4, **extra)
    print(f"{a.out}: {coeff[0].shape[1]} terms, exponents {list(map(int, deg))}")


if __name__ == "__main__":
    main()
