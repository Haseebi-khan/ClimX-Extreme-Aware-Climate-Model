"""Emulator v1: simulated years moved to the target climate, on the GPU.

Each realisation of a target year is a simulated training year whose daily fields are adjusted from the
warming level of its own year to the warming level of the target year. The adjustment comes from a small
per-cell climate model fitted on the training runs:

    temperature (tasmax, tasmin, tas):  T = mu(doy, g) + sigma(doy, g) * z
        mu and sigma^2 are harmonic functions of the day of year whose coefficients are linear in a small
        state vector x of the year (forced warming level g, optionally g^2, aerosol and CO2 terms).
        A source day is moved by keeping its standardised anomaly z:
        T' = mu(doy, x_target) + sigma(doy, x_target) / sigma(doy, x_source) * (T - mu(doy, x_source))
    precipitation:  pr' = pr * m(doy, x_target) / m(doy, x_source), m = fitted seasonal mean (optional)

The 15 indices are computed on every adjusted year with the GPU index code and averaged without weights
over the realisations. With the adjustment switched off this reproduces emulator v0 exactly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from . import data
from .analog import Pool
from .indices import compute_indices
from .paths import INDEX_NAMES

RX = INDEX_NAMES.index("Rx5day")
TEMPS = ("tasmax", "tasmin", "tas")
VARS = TEMPS + ("pr",)


def harmonics(n_harm: int, device) -> torch.Tensor:
    """Seasonal basis [365, 1 + 2 * n_harm]."""
    d = torch.arange(365, device=device, dtype=torch.float64) + 0.5
    cols = [torch.ones_like(d)]
    for k in range(1, n_harm + 1):
        cols += [torch.cos(2 * np.pi * k * d / 365), torch.sin(2 * np.pi * k * d / 365)]
    return torch.stack(cols, dim=1)


class DailyPool:
    """Daily fields of the pool's source years on the GPU: [S, 365, P] per variable (pr in mm/day)."""

    def __init__(self, pool: Pool, cells: np.ndarray, device: str = "cuda", state: np.ndarray | None = None):
        """state [S, J]: state vector of each source year; default is the warming level alone."""
        self.pool, self.cells, self.device = pool, cells, device
        st = pool.gwl[:, None] if state is None else state
        self.state = torch.from_numpy(np.ascontiguousarray(st)).to(device, torch.float64)
        cells_t = torch.from_numpy(cells).to(device)
        self.x = {}
        for var in VARS:
            out = torch.empty((len(pool), 365, len(cells)), dtype=torch.float32, device=device)
            for run in dict.fromkeys(pool.run):
                arr = data.open_var(run, var)
                sel = np.flatnonzero(pool.run == run)
                for a in range(0, len(sel), 16):
                    idx = sel[a:a + 16]
                    r0, r1 = pool.row[idx[0]], pool.row[idx[-1]] + 1
                    assert r1 - r0 == len(idx)
                    block = torch.from_numpy(np.ascontiguousarray(arr[r0 * 365:r1 * 365])).to(device)
                    out[idx[0]:idx[-1] + 1] = block.view(len(idx), 365, -1)[:, :, cells_t]
            self.x[var] = out * 86400 if var == "pr" else out
        self.gwl = torch.from_numpy(pool.gwl).to(device)
        thr = data.load_thresholds()
        self.thr = {k: torch.from_numpy(v[:, cells]).to(device) for k, v in thr.items()}

    def __len__(self) -> int:
        return len(self.pool)


@dataclass
class ClimateModel:
    """Per-cell seasonal mean and variance of the daily fields, linear in the state vector x [J].

    A coefficient array [(1 + J) * K, P] holds the K harmonic coefficients of the value at x = 0 followed
    by the K coefficients of its change per unit of each state component."""
    phi: torch.Tensor                     # [365, K]   basis of the seasonal mean of the temperatures
    phi_v: torch.Tensor                   # [365, Kv]  basis of the variances and of mean precipitation
    mean: dict[str, torch.Tensor]         # variable -> [(1 + J) * K, P]
    var: dict[str, torch.Tensor]          # variable -> [(1 + Jv) * Kv, P] variance of the anomalies
    pr_mean: torch.Tensor | None          # [(1 + Jv) * Kv, P] seasonal mean precipitation (mm/day)
    n_state_var: int = 1                  # Jv: leading state components used by var and pr_mean


def _coef_at(coef: torch.Tensor, k: int, x: torch.Tensor) -> torch.Tensor:
    """Harmonic coefficients at state x. coef [(1 + J) * K, P], x [..., J] -> [..., K, P]."""
    c = coef.view(-1, k, coef.shape[1])                       # [1 + J, K, P]
    j = c.shape[0] - 1
    return c[0] + torch.einsum("...j,jkp->...kp", x[..., :j], c[1:])


@torch.no_grad()
def fit_climate(dp: DailyPool, n_harm: int = 4, n_harm_var: int = 2, n_state_var: int = 1,
                fit_pr: bool = True, ridge: float = 0.0, block: int = 16) -> ClimateModel:
    """Least-squares fit of the seasonal mean (and of the squared residual) on [phi, phi * x_1, ...].

    The variance and precipitation models use only the first ``n_state_var`` state components."""
    dev = dp.device
    phi, phi_v = harmonics(n_harm, dev), harmonics(n_harm_var, dev)
    s = len(dp)

    def design(basis, st):                   # [S, 365, (1 + J) * K]
        parts = [basis[None].expand(s, -1, -1)] + [basis[None] * st[:, j, None, None] for j in range(st.shape[1])]
        return torch.cat(parts, dim=2)

    def solve(basis, st, target_fn):
        f = design(basis, st)
        a = torch.einsum("sdk,sdl->kl", f, f)
        if ridge:
            pen = ridge * torch.diag(a).mean() * torch.eye(a.shape[0], dtype=a.dtype, device=dev)
            pen[:basis.shape[1], :basis.shape[1]] = 0.0
            a = a + pen
        b = torch.zeros((f.shape[2], len(dp.cells)), dtype=torch.float64, device=dev)
        for i in range(0, s, block):
            b += torch.einsum("sdk,sdp->kp", f[i:i + block], target_fn(i, i + block).double())
        return torch.linalg.solve(a, b)

    st_v = dp.state[:, :n_state_var]
    k = phi.shape[1]
    mean, var = {}, {}
    for v in TEMPS:
        mean[v] = solve(phi, dp.state, lambda i, j, v=v: dp.x[v][i:j])

        def resid2(i, j, v=v):
            mu = torch.einsum("dk,skp->sdp", phi, _coef_at(mean[v], k, dp.state[i:j]))
            return (dp.x[v][i:j].double() - mu) ** 2
        var[v] = solve(phi_v, st_v, resid2)
    pr_mean = solve(phi_v, st_v, lambda i, j: dp.x["pr"][i:j]) if fit_pr else None
    return ClimateModel(phi=phi, phi_v=phi_v, mean=mean, var=var, pr_mean=pr_mean, n_state_var=n_state_var)


@dataclass
class Transform:
    """What is adjusted when a source year is moved to the target warming level."""
    shift: bool = True          # move the seasonal mean of the temperatures
    scale: bool = False         # rescale the temperature anomalies by the ratio of standard deviations
    pr: bool = False            # scale precipitation by the ratio of seasonal means
    max_ratio: float = 1.5      # bounds on the scale factors (ratio in [1 / max_ratio, max_ratio])
    pr_max_ratio: float = 2.0


@torch.no_grad()
def emulate(dp: DailyPool, cm: ClimateModel | None, tr: Transform | None, target_state: np.ndarray,
            counts: np.ndarray, last_year_short: bool = True, batch: int = 24,
            progress: bool = False) -> torch.Tensor:
    """Mean indices over realisations for every target year: [T, 15, P].

    target_state [T, J] (or [T] for the warming level alone): state vector of each target year.
    counts [T, S]: number of realisations of target year t that use source year s (rows sum to N).
    Realisations are paired across years in sorted order (as in ``resample.sequences_from_counts``).
    """
    dev = dp.device
    n_t, s = counts.shape
    n = int(counts[0].sum())
    out = []
    prev_tail = None           # [S_prev_distinct, 4, P] adjusted last four days of the previous year's sources
    prev_src = None
    ts = torch.from_numpy(np.asarray(target_state, dtype=np.float64).reshape(n_t, -1)).to(dev)
    if cm is not None:
        k, kv, jv = cm.phi.shape[1], cm.phi_v.shape[1], cm.n_state_var

        def field(basis, coef, kk, x):          # seasonal field(s) at state x: [..., 365, P]
            return torch.einsum("dk,...kp->...dp", basis, _coef_at(coef, kk, x)).float()
    for t in range(n_t):
        short = last_year_short and t == n_t - 1
        nd = 364 if short else 365
        src = np.flatnonzero(counts[t])
        cnt = torch.from_numpy(counts[t][src]).to(dev, torch.float32)
        use_tr = tr is not None and cm is not None
        if use_tr:
            mu_t = {v: field(cm.phi, cm.mean[v], k, ts[t]) for v in TEMPS} if (tr.shift or tr.scale) else None
            sg_t = ({v: field(cm.phi_v, cm.var[v], kv, ts[t, :jv]).clamp(min=1e-4).sqrt() for v in TEMPS}
                    if tr.scale else None)
            pm_t = field(cm.phi_v, cm.pr_mean, kv, ts[t, :jv]).clamp(min=1e-3) if tr.pr else None
        acc = torch.zeros((len(INDEX_NAMES), len(dp.cells)), dtype=torch.float64, device=dev)
        rx_in = torch.empty((len(src), len(dp.cells)), dtype=torch.float32, device=dev)
        head = torch.empty((len(src), 4, len(dp.cells)), dtype=torch.float32, device=dev)
        tail = torch.empty((len(src), 4, len(dp.cells)), dtype=torch.float32, device=dev)
        for a in range(0, len(src), batch):
            ids = torch.from_numpy(src[a:a + batch]).to(dev)
            x = {v: dp.x[v][ids] for v in VARS}                       # [B, 365, P]
            if use_tr:
                xs = dp.state[ids]
                for v in TEMPS:
                    if not (tr.shift or tr.scale):
                        continue
                    mu_s = field(cm.phi, cm.mean[v], k, xs)
                    if tr.scale:
                        s2 = field(cm.phi_v, cm.var[v], kv, xs[:, :jv])
                        ratio = (sg_t[v][None] / s2.clamp(min=1e-4).sqrt()).clamp(1 / tr.max_ratio, tr.max_ratio)
                        x[v] = mu_t[v][None] + ratio * (x[v] - mu_s)
                    else:
                        x[v] = x[v] + (mu_t[v][None] - mu_s)
                if tr.shift or tr.scale:
                    lo = torch.minimum(x["tasmin"], x["tasmax"])
                    hi = torch.maximum(x["tasmin"], x["tasmax"])
                    x["tasmin"], x["tasmax"] = lo, hi
                    x["tas"] = torch.minimum(torch.maximum(x["tas"], lo), hi)
                if tr.pr:
                    pm_s = field(cm.phi_v, cm.pr_mean, kv, xs[:, :jv])
                    x["pr"] = x["pr"] * (pm_t[None] / pm_s.clamp(min=1e-3)).clamp(1 / tr.pr_max_ratio, tr.pr_max_ratio)
            idx = compute_indices(x["tasmax"][:, :nd], x["tasmin"][:, :nd], x["tas"][:, :nd], x["pr"][:, :nd],
                                  dp.thr)                                          # [B, 15, P]
            acc += torch.einsum("b,bip->ip", cnt[a:a + batch].double(), idx.double())
            rx_in[a:a + batch] = idx[:, RX]
            head[a:a + batch] = x["pr"][:, :4]
            tail[a:a + batch] = x["pr"][:, 361:365]
            del x, idx
        mean = (acc / n).float()
        # Rx5day per realisation, with the windows that straddle the previous year
        pos = torch.from_numpy(np.repeat(np.arange(len(src)), counts[t][src])).to(dev)      # [N] -> row in src
        rx = rx_in[pos]
        if t > 0:
            ppos = torch.from_numpy(np.repeat(np.arange(len(prev_src)), counts[t - 1][prev_src])).to(dev)
            for j in range(4):
                rx = torch.maximum(rx, prev_tail[ppos][:, j:].sum(dim=1) + head[pos][:, :j + 1].sum(dim=1))
        mean[RX] = rx.mean(dim=0)
        out.append(mean)
        prev_tail, prev_src = tail, src
        if progress and (t % 10 == 0 or t == n_t - 1):
            print(f"   year {t + 1}/{n_t}: {len(src)} source years", flush=True)
    return torch.stack(out)
