"""Access to the flat daily store built by scripts/build_store.py, the AR6 land mask and the thresholds."""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

from .paths import WORK, NLAT, NLON
from . import official_rnNSE

STORE = WORK / "store"
DAYS = 365


@lru_cache(maxsize=1)
def meta() -> dict:
    return json.loads((STORE / "meta.json").read_text())


def open_var(scenario: str, var: str) -> np.memmap:
    """Raw daily array [n_days, 192*288] (float32, read-only)."""
    n = meta()["n_days"][scenario]
    return np.memmap(STORE / scenario / f"{var}.f32", dtype=np.float32, mode="r", shape=(n, NLAT * NLON))


def start_year(scenario: str) -> int:
    return meta()["start_year"][scenario]


def n_years(scenario: str) -> int:
    """Number of calendar years, the last one being one day short (364 days)."""
    return -(-meta()["n_days"][scenario] // DAYS)


def year_slice(scenario: str, year: int) -> slice:
    """Day range of a calendar year inside the scenario array."""
    i = year - start_year(scenario)
    n = meta()["n_days"][scenario]
    if i < 0 or i * DAYS >= n:
        raise IndexError(f"{year} is outside {scenario}")
    return slice(i * DAYS, min((i + 1) * DAYS, n))


@lru_cache(maxsize=1)
def region_mask() -> np.ndarray:
    """AR6 land-region code per cell [192, 288]; -1 = not scored (taken from the official metric)."""
    mask, _ = official_rnNSE._decode_mask()
    return mask


@lru_cache(maxsize=1)
def land_index() -> np.ndarray:
    """Flat indices of the scored (AR6 land) cells, in increasing order."""
    return np.flatnonzero(region_mask().ravel() >= 0)


@lru_cache(maxsize=1)
def lats() -> np.ndarray:
    return np.asarray(meta()["lat"], dtype=np.float64)


def _floor32(x: np.ndarray) -> np.ndarray:
    """Largest float32 that is <= x."""
    y = x.astype(np.float32)
    return np.where(y.astype(np.float64) > x, np.nextafter(y, np.float32(-np.inf)), y)


def _ceil32(x: np.ndarray) -> np.ndarray:
    """Smallest float32 that is >= x."""
    y = x.astype(np.float32)
    return np.where(y.astype(np.float64) < x, np.nextafter(y, np.float32(np.inf)), y)


def load_thresholds() -> dict[str, np.ndarray]:
    """Day-of-year thresholds [365, 192*288] in float32: tx90, tn10 (K) and pr95 (mm/day).

    The official code compares float32 data with float64 thresholds. For float32 data ``x > t`` equals
    ``x > floor32(t)`` and ``x < t`` equals ``x < ceil32(t)``, so the thresholds are rounded in the
    direction that keeps every comparison identical: down for tx90 and pr95 (used with >), up for tn10
    (used with <)."""
    z = np.load(WORK / "thresholds.npz")
    return {"tx90": _floor32(z["tx90"]), "tn10": _ceil32(z["tn10"]), "pr95": _floor32(z["pr95"])}


def load_years(scenario: str, var: str, cells: np.ndarray | None = None,
               years: range | None = None) -> np.ndarray:
    """Daily values as [n_years, 365, n_cells]. Only complete years are returned (the truncated last
    year of each run is dropped unless ``years`` names it explicitly, in which case it is NaN-padded)."""
    arr = open_var(scenario, var)
    y0 = start_year(scenario)
    if years is None:
        years = range(y0, y0 + arr.shape[0] // DAYS)
    ncell = arr.shape[1] if cells is None else len(cells)
    out = np.full((len(years), DAYS, ncell), np.nan, dtype=np.float32)
    for k, y in enumerate(years):
        sl = year_slice(scenario, y)
        block = arr[sl] if cells is None else arr[sl][:, cells]
        out[k, :block.shape[0]] = block
    return out
