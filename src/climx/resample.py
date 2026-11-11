"""Emulator v0: every realisation of a target year is a whole simulated year taken from the training runs.

A realisation is a daily series for 2015-2100 made of 86 simulated years placed one after another (the
last one cut to 364 days, as the target period is). The indices of such a series are those of its years,
except for the 5-day precipitation windows that straddle two years, which are recomputed for each pair of
consecutive years. The functions here return the unweighted mean of the indices over the realisations,
exactly as if the daily series had been written out and passed through the index code.
"""
from __future__ import annotations

import numpy as np
import torch

from .paths import WORK, INDEX_NAMES
from .analog import Pool

RX = INDEX_NAMES.index("Rx5day")


def load_tables(pool: Pool, cells: np.ndarray | None, device: str = "cuda") -> dict[str, torch.Tensor]:
    """Index tables of the pool's source years, in pool order: idx365/idx364 [S, 15, C], first4/last4 [S, 4, C]."""
    out = {k: [] for k in ("idx365", "idx364", "first4", "last4")}
    for run in dict.fromkeys(pool.run):
        z = np.load(WORK / "tables" / f"{run}.npz")
        rows = pool.row[pool.run == run]
        for k in out:
            a = z[k][rows]
            out[k].append(torch.from_numpy(a if cells is None else a[:, :, cells]))
    return {k: torch.cat(v).to(device) for k, v in out.items()}


def sequences_from_counts(counts: np.ndarray) -> np.ndarray:
    """counts [T, S] (rows sum to N) -> source index of each realisation in each year [T, N], sorted."""
    n = int(counts[0].sum())
    seq = np.empty((counts.shape[0], n), dtype=np.int64)
    for t in range(counts.shape[0]):
        assert counts[t].sum() == n
        seq[t] = np.repeat(np.arange(counts.shape[1]), counts[t])
    return seq


@torch.no_grad()
def emulate(seq: np.ndarray, tab: dict[str, torch.Tensor], last_year_short: bool = True) -> torch.Tensor:
    """Mean indices over realisations. seq [T, N] -> [T, 15, C] (float32)."""
    dev = tab["idx365"].device
    n_t, n = seq.shape
    out = []
    for t in range(n_t):
        short = last_year_short and t == n_t - 1
        table = tab["idx364"] if short else tab["idx365"]
        cur = torch.from_numpy(seq[t]).to(dev)
        src, cnt = torch.unique(cur, return_counts=True)
        mean = torch.einsum("s,sic->ic", cnt.to(torch.float32) / n, table[src])
        rx = table[cur, RX]                                           # [N, C] windows inside the year
        if t > 0:
            prev = torch.from_numpy(seq[t - 1]).to(dev)
            tail, head = tab["last4"][prev], tab["first4"][cur]       # [N, 4, C]
            for j in range(4):                                        # window ending on day j+1 of the year
                rx = torch.maximum(rx, tail[:, j:].sum(dim=1) + head[:, :j + 1].sum(dim=1))
        mean[RX] = rx.mean(dim=0)
        out.append(mean)
    return torch.stack(out)
