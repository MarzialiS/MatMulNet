"""Reference PyTorch implementation of MatMulNet and of one training epoch.

It is the direct transcription of the model and of the training step: plain autograd,
torch.nn.utils.clip_grad_norm_ and torch.optim.Adam. `train.py --backend torch` runs it;
the default backend (`kernels.py`) computes the same quantities in compiled code.
"""
import numpy as np
import torch
from torch import nn

torch.set_default_dtype(torch.float64)


class MatMulNet(nn.Module):
    """Bilinear network with r multiplications for n x n matrices.

    Parameters are coefficient tensors of shape (n^2, r, d), (n^2, r, d) and (r, n^2, d); the
    factors are H(eps) = sum_e eps^e H_coeff[:, :, e - d_min], and likewise K and F. With the
    single exponent 0 (the default) this is the rank model, with factors H, K, F themselves."""

    def __init__(self, n, r, degrees=(0,), alpha=1.0, generator=None):
        super().__init__()
        m, d = n * n, len(degrees)
        self.register_buffer("degrees", torch.tensor(degrees, dtype=torch.float64))
        self.register_buffer("eps", torch.tensor(1.0))
        self.H_coeff = nn.Parameter(alpha * torch.randn(m, r, d, generator=generator))
        self.K_coeff = nn.Parameter(alpha * torch.randn(m, r, d, generator=generator))
        self.F_coeff = nn.Parameter(alpha * torch.randn(r, m, d, generator=generator))

    def factors(self):
        tp = self.eps ** self.degrees
        return self.H_coeff @ tp, self.K_coeff @ tp, self.F_coeff @ tp

    def forward(self, a, b):
        """a, b: batches of flattened matrices, shape (batch, n^2). Returns vec(AB) predictions."""
        H, K, F = self.factors()
        return ((a @ H) * (b @ K)) @ F


class TorchBackend:
    def __init__(self, params, degrees, clip):
        n = int(round(np.sqrt(params[0].shape[0])))
        self.net = MatMulNet(n, params[0].shape[1], tuple(int(d) for d in degrees))
        self.clip = clip
        self.opt = torch.optim.Adam(self.net.parameters(), lr=1.0)
        self.loss = nn.MSELoss()
        self.set_state(params, None, None, 0.0)

    def _params(self):
        return [self.net.H_coeff, self.net.K_coeff, self.net.F_coeff]

    def set_state(self, params, m, v, step):
        with torch.no_grad():
            for p, x in zip(self._params(), params):
                p.copy_(torch.from_numpy(np.asarray(x)))
        if step > 0:
            self.opt.state.clear()
            for p, mi, vi in zip(self._params(), m, v):
                self.opt.state[p] = {"step": torch.tensor(float(step)),
                                     "exp_avg": torch.from_numpy(mi.copy()),
                                     "exp_avg_sq": torch.from_numpy(vi.copy())}

    def get_state(self):
        P = [p.detach().numpy().copy() for p in self._params()]
        st = [self.opt.state.get(p) for p in self._params()]
        if st[0] is None:
            return P, [np.zeros_like(p) for p in P], [np.zeros_like(p) for p in P], 0.0
        return (P, [s["exp_avg"].numpy().copy() for s in st],
                [s["exp_avg_sq"].numpy().copy() for s in st], float(st[0]["step"]))

    def train_epoch(self, eps, lr, data, perm, batch):
        A, B, Y = (torch.from_numpy(x) for x in data)
        self.net.eps.fill_(eps)
        for g in self.opt.param_groups:
            g["lr"] = lr
        total = 0.0
        for start in range(0, len(perm), batch):
            idx = torch.from_numpy(perm[start:start + batch])
            self.opt.zero_grad()
            loss = self.loss(self.net(A[idx], B[idx]), Y[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=self.clip)
            self.opt.step()
            total += loss.item() * len(idx)
        return total / len(perm)

    @torch.no_grad()
    def validate(self, eps, data):
        A, B, Y = (torch.from_numpy(x) for x in data)
        self.net.eps.fill_(eps)
        return self.loss(self.net(A, B), Y).item()
