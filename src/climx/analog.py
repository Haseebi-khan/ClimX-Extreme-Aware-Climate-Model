"""Shared pieces of the analog-year emulators: the pool of simulated years, their warming levels,
kernel weights, and the conversion of weights into an integer number of draws per source year."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .paths import WORK
from .forcing import GmstEmulator, forcing_table

SSPS = ("ssp126", "ssp370", "ssp585")
RUNS = ("historical",) + SSPS
PREINDUSTRIAL_YEARS = 50      # 1850-1899 mean defines the zero of the warming level


def annual_gmst() -> dict[str, np.ndarray]:
    ann = np.load(WORK / "annual.npz")
    return {s: ann[f"{s}_gmst"] for s in RUNS}


def polyfit_smooth(y: np.ndarray, deg: int) -> np.ndarray:
    x = np.linspace(-1, 1, len(y))
    return np.polyval(np.polyfit(x, y, deg), x)


def gauss_smooth(y: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian low-pass with renormalisation at the ends."""
    n = len(y)
    i = np.arange(n)
    w = np.exp(-0.5 * ((i[:, None] - i[None, :]) / sigma) ** 2)
    return (w @ y) / w.sum(axis=1)


@dataclass
class Pool:
    """Source years available to an emulator."""
    run: np.ndarray        # [S] run name of each source year
    year: np.ndarray       # [S] calendar year
    row: np.ndarray        # [S] row inside the run's table
    gwl: np.ndarray        # [S] forced warming level (K above 1850-1899)

    def __len__(self) -> int:
        return len(self.year)


def source_gwl(gmst: dict[str, np.ndarray], method: str = "poly") -> dict[str, np.ndarray]:
    """Forced warming level of every training year, from the run's own global-mean temperature.

    poly   : polynomial fit in time per run (degree 6 for the historical run, 3 for the SSPs)
    joint  : Gaussian low-pass (sigma 12 years) over historical + SSP as one 251-year series, so that the
             three SSPs share the estimate at the 2015 branch point
    """
    pre = gmst["historical"][:PREINDUSTRIAL_YEARS].mean()
    out = {}
    if method == "poly":
        out["historical"] = polyfit_smooth(gmst["historical"], 6) - pre
        for s in SSPS:
            out[s] = polyfit_smooth(gmst[s], 3) - pre
    elif method == "joint":
        hist_parts = []
        for s in SSPS:
            full = gauss_smooth(np.concatenate([gmst["historical"], gmst[s]]), 12.0) - pre
            out[s] = full[165:]
            hist_parts.append(full[:165])
        out["historical"] = np.mean(hist_parts, axis=0)
    else:
        raise ValueError(method)
    return out


def build_pool(runs: tuple[str, ...], gwl: dict[str, np.ndarray]) -> Pool:
    run, year, row, g = [], [], [], []
    for r in runs:
        y0 = 1850 if r == "historical" else 2015
        n = len(gwl[r]) - 1                       # complete years only (the last year is one day short)
        run += [r] * n
        year += list(range(y0, y0 + n))
        row += list(range(n))
        g += list(gwl[r][:n])
    return Pool(np.array(run), np.array(year), np.array(row), np.array(g, dtype=np.float64))


def target_gwl(scenario: str, gmst: dict[str, np.ndarray], train_ssps: tuple[str, ...],
               emulator: GmstEmulator | None = None) -> np.ndarray:
    """Emulated forced warming level of a scenario for 2015-2100, fitted without that scenario's output."""
    em = (emulator or GmstEmulator(taus=(8.0,), use=(0,), ridge=1e-2)).fit(gmst, train_ssps)
    pre = gmst["historical"][:PREINDUSTRIAL_YEARS].mean()
    return em.predict(scenario)[165:] - pre


def gaussian_weights(g_target: np.ndarray, g_source: np.ndarray, h: float) -> np.ndarray:
    """Row-normalised Gaussian kernel weights [T, S]."""
    d = (g_source[None, :] - g_target[:, None]) / h
    w = np.exp(-0.5 * d * d)
    return w / w.sum(axis=1, keepdims=True)


def systematic_counts(w: np.ndarray, n_draws: int) -> np.ndarray:
    """Turn each row of weights into integer draw counts that sum to n_draws (systematic resampling with a
    fixed offset of one half, so the result is deterministic)."""
    cum = np.cumsum(w, axis=1)
    cum = cum / cum[:, -1:]
    edges = np.floor(cum * n_draws + 0.5)
    counts = np.diff(np.concatenate([np.zeros((w.shape[0], 1)), edges], axis=1), axis=1)
    return counts.astype(np.int64)


STATE_FEATURES = ("g", "g2", "so2", "co2", "bc", "ch4")


def forcing_feature(run: str, name: str) -> np.ndarray:
    """Annual forcing feature of a run over its own years (165 for historical, 86 for an SSP)."""
    tab = forcing_table()
    ref = {v: tab[f"historical_{v}"][0] for v in ("SO2", "CO2_LBC", "CH4_LBC")}
    ref_bc = tab["historical_BC_AX"][0] + tab["historical_BC_N"][0]
    if name == "so2":
        return np.log(tab[f"{run}_SO2"] / ref["SO2"])
    if name == "co2":
        return np.log(tab[f"{run}_CO2_LBC"] / ref["CO2_LBC"])
    if name == "ch4":
        return np.log(tab[f"{run}_CH4_LBC"] / ref["CH4_LBC"])
    if name == "bc":
        return np.log((tab[f"{run}_BC_AX"] + tab[f"{run}_BC_N"]) / ref_bc)
    raise ValueError(name)


def state_matrix(run: np.ndarray, year: np.ndarray, gwl: np.ndarray, features: tuple[str, ...]) -> np.ndarray:
    """State vector of each (run, year): columns in the order of ``features``; "g" must come first."""
    assert features[0] == "g"
    cols = []
    for f in features:
        if f == "g":
            cols.append(gwl)
        elif f == "g2":
            cols.append(gwl ** 2)
        else:
            col = np.empty(len(year))
            for r in np.unique(run):
                m = run == r
                y0 = 1850 if r == "historical" else 2015
                col[m] = forcing_feature(str(r), f)[year[m] - y0]
            cols.append(col)
    return np.stack(cols, axis=1)
