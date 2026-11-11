"""Region-wise nNSE (the leaderboard metric) on arrays.

Mirrors the organisers' ``rnNSE.py`` (kept unchanged as ``official_rnNSE.py``): per index and scored cell,
R2 = 1 - MSE / Var_t(truth) with the population variance, nNSE = R2 / (2 - R2); cells with
Var_t(truth) <= 1e-6 are dropped; cosine-latitude weighted mean inside each AR6 land region, then a plain
mean over regions and over the 15 indices. ``tests/test_metric.py`` checks it against the official function.
"""
from __future__ import annotations

import numpy as np
import torch

from . import data

VAR_EPS = 1e-6


class Scorer:
    """Scores predictions given for the scored (AR6 land) cells only.

    Arrays are [T, n_index, n_land] with the land cells in the order of ``data.land_index()``.
    With ``cells`` (flat grid indices of a subset of the scored cells) the score is computed on that
    subset only, which is used to speed up parameter sweeps.
    """

    def __init__(self, device: str = "cuda", cells: np.ndarray | None = None):
        self.device = device
        land = data.land_index() if cells is None else np.asarray(cells)
        region = data.region_mask().ravel()[land]
        lat_idx = land // data.region_mask().shape[1]
        cosw = np.cos(np.deg2rad(data.lats()))[lat_idx]
        self.codes = np.unique(region)
        self.region = torch.from_numpy(np.searchsorted(self.codes, region)).to(device)        # [P]
        self.cosw = torch.from_numpy(cosw).to(device)                                          # [P] float64
        self.n_region = len(self.codes)

    @torch.no_grad()
    def cell_stats(self, truth: torch.Tensor):
        """Per-cell temporal variance of the truth and the validity mask. truth: [T, I, P]."""
        t = truth.to(self.device, torch.float64)
        var = t.var(dim=0, unbiased=False)
        return var, var > VAR_EPS

    @torch.no_grad()
    def score(self, truth: torch.Tensor, pred: torch.Tensor, detail: bool = False):
        """Leaderboard score. Returns S, or (S, per-index scores [I], region table [I, R]) with detail."""
        t = truth.to(self.device, torch.float64)
        p = pred.to(self.device, torch.float64)
        var = t.var(dim=0, unbiased=False)                    # [I, P]
        mse = ((t - p) ** 2).mean(dim=0)
        valid = var > VAR_EPS
        r2 = 1.0 - mse / var.clamp(min=VAR_EPS)
        nnse = torch.where(valid, r2 / (2.0 - r2), torch.zeros_like(r2))
        w = self.cosw[None, :] * valid                         # [I, P]
        n_i = t.shape[1]
        num = torch.zeros(n_i, self.n_region, dtype=torch.float64, device=self.device)
        den = torch.zeros_like(num)
        idx = self.region[None, :].expand(n_i, -1)
        num.scatter_add_(1, idx, nnse * w)
        den.scatter_add_(1, idx, w)
        ok = den >= 1e-12
        reg = torch.where(ok, num / den.clamp(min=1e-12), torch.full_like(num, float("nan")))
        s_v = torch.nansum(reg, dim=1) / ok.sum(dim=1).clamp(min=1)
        s = float(s_v.mean())
        if detail:
            return s, s_v.cpu().numpy(), reg.cpu().numpy()
        return s
