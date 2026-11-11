"""The 15 ClimX extreme indices on the GPU.

A PyTorch reimplementation of the organisers' ``src/indices_xclim.py`` (xclim 0.59.1) followed by
``format_indices`` (annual max/min of the monthly TXx, TNn and Rx5day). It works on blocks of whole
years and reproduces the official code value for value (see ``scripts/verify_indices.py``).

Conventions
-----------
* Daily arrays have shape ``[..., D, P]``: D days of one calendar year (365, or 364 for the truncated
  last year of each file) and P grid cells. Leading dimensions are free (years, realisations).
* Temperatures are in K, precipitation in mm/day (``pr_kg_m2_s * 86400`` in float32, as the official
  code does).
* ``thr`` holds the day-of-year thresholds computed by xclim's ``percentile_doy`` on the historical
  1961-1990 base period: ``tx90`` and ``tn10`` in K, ``pr95`` in mm/day, each ``[365, P]``.
"""
from __future__ import annotations

import torch

INDEX_NAMES = ("TXx", "TNn", "SU", "TR", "FD", "ID", "WSDI", "CSDI", "GSL",
               "Rx5day", "CDD", "CWD", "R95p", "R10mm", "SDII")
MID_IDX = 181          # 1 July in a 365-day calendar (0-based day of year)
SPELL_WINDOW = 6       # WSDI / CSDI / growing season window


def run_length_fwd(mask: torch.Tensor) -> torch.Tensor:
    """Length of the run of True values ending at each day (0 where False). mask: [..., D, P]."""
    c = torch.cumsum(mask, dim=-2, dtype=torch.int16)
    reset = torch.where(mask, torch.zeros_like(c), c)
    reset = torch.cummax(reset, dim=-2).values
    return c - reset


def run_length_bwd(mask: torch.Tensor) -> torch.Tensor:
    """Length of the run of True values starting at each day (0 where False)."""
    return run_length_fwd(mask.flip(-2)).flip(-2)


def longest_run(mask: torch.Tensor) -> torch.Tensor:
    return run_length_fwd(mask).amax(dim=-2)


def windowed_run_count(mask: torch.Tensor, window: int = SPELL_WINDOW) -> torch.Tensor:
    """Number of days that belong to runs of at least ``window`` consecutive True days."""
    fwd = run_length_fwd(mask)
    nxt = torch.zeros_like(mask)
    nxt[..., :-1, :] = mask[..., 1:, :]
    is_end = mask & ~nxt
    keep = is_end & (fwd >= window)
    return (fwd * keep).sum(dim=-2, dtype=torch.int32)


def growing_season_length(tas: torch.Tensor, thresh: float = 278.15,
                          window: int = SPELL_WINDOW, mid_idx: int = MID_IDX) -> torch.Tensor:
    """xclim ``growing_season_length`` (thresh 5 degC, window 6, mid_date 07-01, freq YS, op >=)."""
    d = tas.shape[-2]
    da = tas >= thresh
    idx = torch.arange(d, device=tas.device).view(*([1] * (tas.dim() - 2)), d, 1)
    big = d + 10
    # start: first day of the first run of >= window True days that lies before mid_idx + window - 1
    start_ok = (run_length_bwd(da) >= window) & (idx < mid_idx)
    beg = torch.where(start_ok, idx, big).amin(dim=-2)
    has_beg = beg < big
    # end: first day (>= mid date) of a run of >= window False days
    end_ok = (run_length_bwd(~da) >= window) & (idx >= mid_idx)
    end = torch.where(end_ok, idx, big).amin(dim=-2)
    has_end = end < big
    length = torch.where(has_end, end - beg, d - beg)
    return torch.where(has_beg, length, torch.zeros_like(length))


def rolling_sum5(pr: torch.Tensor, prev4: torch.Tensor | None) -> torch.Tensor:
    """Right-aligned 5-day sums. With ``prev4`` (the 4 days before the block) every day gets a value;
    without it the first 4 days are dropped (they are NaN in the official code)."""
    x = pr if prev4 is None else torch.cat([prev4, pr], dim=-2)
    n = x.shape[-2]
    return (x[..., 0:n - 4, :] + x[..., 1:n - 3, :] + x[..., 2:n - 2, :]
            + x[..., 3:n - 1, :] + x[..., 4:n, :])


def _doy_slice(t: torch.Tensor, d: int) -> torch.Tensor:
    return t[:d]


@torch.no_grad()
def compute_indices(tasmax: torch.Tensor, tasmin: torch.Tensor, tas: torch.Tensor, pr: torch.Tensor,
                    thr: dict[str, torch.Tensor], pr_prev4: torch.Tensor | None = None,
                    names: tuple[str, ...] = INDEX_NAMES) -> torch.Tensor:
    """All requested indices for a block of years.

    Inputs ``[..., D, P]`` (float32). Returns ``[..., len(names), P]`` float32.
    ``pr_prev4`` is ``[..., 4, P]``: the last four days before each year, or None for a year that
    starts the series.
    """
    d = tasmax.shape[-2]
    out = {}
    want = set(names)
    if "TXx" in want:
        out["TXx"] = tasmax.amax(dim=-2)
    if "TNn" in want:
        out["TNn"] = tasmin.amin(dim=-2)
    if "SU" in want:
        out["SU"] = (tasmax > 298.15).sum(dim=-2)
    if "TR" in want:
        out["TR"] = (tasmin > 293.15).sum(dim=-2)
    if "FD" in want:
        out["FD"] = (tasmin < 273.15).sum(dim=-2)
    if "ID" in want:
        out["ID"] = (tasmax < 273.15).sum(dim=-2)
    if "WSDI" in want:
        out["WSDI"] = windowed_run_count(tasmax > _doy_slice(thr["tx90"], d))
    if "CSDI" in want:
        out["CSDI"] = windowed_run_count(tasmin < _doy_slice(thr["tn10"], d))
    if "GSL" in want:
        out["GSL"] = growing_season_length(tas)
    if "Rx5day" in want:
        out["Rx5day"] = rolling_sum5(pr, pr_prev4).amax(dim=-2)
    if "CDD" in want:
        out["CDD"] = longest_run(pr < 1.0)
    if "CWD" in want:
        out["CWD"] = longest_run(pr > 1.0)
    if "R95p" in want:
        out["R95p"] = torch.where(pr > _doy_slice(thr["pr95"], d), pr, torch.zeros_like(pr)).sum(dim=-2)
    if "R10mm" in want:
        out["R10mm"] = (pr >= 10.0).sum(dim=-2)
    if "SDII" in want:
        wet = pr >= 1.0
        cnt = wet.sum(dim=-2)
        tot = torch.where(wet, pr, torch.zeros_like(pr)).sum(dim=-2)
        out["SDII"] = torch.where(cnt > 0, tot / cnt.clamp(min=1), torch.zeros_like(tot))
    return torch.stack([out[n].to(torch.float32) for n in names], dim=-2)
