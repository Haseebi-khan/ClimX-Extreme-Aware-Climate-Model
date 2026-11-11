"""Forcing series of every scenario and an emulator of the forced global-mean temperature (GMST).

The emulator is a small linear impulse-response model: annual global forcing proxies (log CO2, sqrt CH4,
halocarbons, aerosol emissions) are passed through exponential filters with a fast and a slow time scale and
combined linearly. Every coefficient is fitted on the ClimX training runs only (historical + training SSPs);
no externally calibrated parameter is used. It gives the smooth, forced warming level of a scenario from its
forcings alone, which is what the test scenario provides.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

from .paths import HF_FULL, WORK

HIST_FILE = "NorESM2-MM_r1i1p1f1_with_forcing_hist_train_1850_1931.nc"
ALL_FILE = "NorESM2-MM_r1i1p1f1_with_forcing_forcings_all.nc"
GHG = ("CO2_LBC", "CH4_LBC", "N2O_LBC", "CFC11eq_LBC", "CF2CL2_LBC")
AER = ("SO2", "BC_AX", "BC_N", "OM_NI")
ALL_SSPS = ("ssp126", "ssp245", "ssp370", "ssp585")


def _global_annual(ds, var: str, k: int) -> np.ndarray:
    lat = np.asarray(ds["lat"][:], dtype=np.float64)
    w = np.cos(np.deg2rad(lat))
    w = w / w.sum() / len(ds.dimensions["lon"])
    x = np.asarray(ds[var][k], dtype=np.float64)
    gm = (x * w[None, :, None]).sum(axis=(1, 2))
    return gm.reshape(-1, 12).mean(axis=1)


def build_forcing_table() -> dict[str, np.ndarray]:
    """Annual global forcing series per scenario: <scen>_year and <scen>_<agent>."""
    import netCDF4
    out = {}
    with netCDF4.Dataset(HF_FULL / HIST_FILE) as ds:
        out["historical_year"] = np.arange(1850, 2015)
        for v in GHG:
            out[f"historical_{v}"] = np.asarray(ds[v][0], dtype=np.float64)
        for v in AER:
            out[f"historical_{v}"] = _global_annual(ds, v, 0)
    with netCDF4.Dataset(HF_FULL / ALL_FILE) as ds:
        scen = [str(s) for s in ds["forcing_scenario"][:]]
        for k, s in enumerate(scen):
            out[f"{s}_year"] = np.arange(2015, 2101)
            for v in GHG:
                out[f"{s}_{v}"] = np.asarray(ds[v][k], dtype=np.float64)
            for v in AER:
                out[f"{s}_{v}"] = _global_annual(ds, v, k)
    return out


@lru_cache(maxsize=1)
def forcing_table() -> dict[str, np.ndarray]:
    path = WORK / "forcings.npz"
    if not path.exists():
        tab = build_forcing_table()
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, **tab)
        os.replace(tmp, path)
    z = np.load(path)
    return {k: z[k] for k in z.files}


def proxies(scenario: str) -> np.ndarray:
    """Forcing proxies for 1850-2100 along one scenario (historical, then the SSP): [251, n_proxy]."""
    tab = forcing_table()

    def series(v):
        return np.concatenate([tab[f"historical_{v}"], tab[f"{scenario}_{v}"]])

    co2, ch4, n2o = series("CO2_LBC"), series("CH4_LBC"), series("N2O_LBC")
    cfc11, cfc12 = series("CFC11eq_LBC"), series("CF2CL2_LBC")
    so2, bc, om = series("SO2"), series("BC_AX") + series("BC_N"), series("OM_NI")
    cols = [np.log(co2 / co2[0]),
            np.sqrt(ch4 * 1e9) - np.sqrt(ch4[0] * 1e9),
            np.sqrt(n2o * 1e9) - np.sqrt(n2o[0] * 1e9),
            (cfc11 - cfc11[0]) * 1e9,
            cfc12 * 1e9,
            np.log(so2 / so2[0]),
            np.log(bc / bc[0]),
            np.log(om / om[0])]
    return np.stack(cols, axis=1)


PROXY_NAMES = ("lnCO2", "sqrtCH4", "sqrtN2O", "CFC11eq", "CFC12", "lnSO2", "lnBC", "lnOM")


def ema(x: np.ndarray, tau: float) -> np.ndarray:
    """Exponential response along axis 0 (a one-box relaxation with time scale tau years)."""
    a = 1.0 - np.exp(-1.0 / tau)
    y = np.zeros_like(x)
    acc = np.zeros(x.shape[1:])
    for i in range(x.shape[0]):
        acc = acc + a * (x[i] - acc)
        y[i] = acc
    return y


class GmstEmulator:
    """T(t) = c0 + sum_j sum_k a_jk * EMA_tau_j[proxy_k](t), ridge-fitted on annual GMST."""

    def __init__(self, taus=(4.0, 150.0), use=(0, 1, 3, 5), ridge: float = 1e-3):
        self.taus, self.use, self.ridge = tuple(taus), tuple(use), ridge
        self.coef = None
        self.scale = None

    def features(self, scenario: str) -> np.ndarray:
        p = proxies(scenario)[:, self.use]
        return np.concatenate([ema(p, tau) for tau in self.taus], axis=1)   # [251, n]

    def fit(self, gmst: dict[str, np.ndarray], scenarios: tuple[str, ...]) -> "GmstEmulator":
        """gmst: annual GMST per run (historical has 165 values, SSPs 86). The historical run is always used."""
        X, y = [], []
        first = True
        for s in scenarios:
            f = self.features(s)
            if first:
                X.append(f[:165]); y.append(gmst["historical"]); first = False
            X.append(f[165:]); y.append(gmst[s])
        X, y = np.concatenate(X), np.concatenate(y)
        self.scale = X.std(axis=0) + 1e-12
        Z = np.concatenate([np.ones((len(X), 1)), X / self.scale], axis=1)
        pen = self.ridge * len(X) * np.eye(Z.shape[1]); pen[0, 0] = 0.0
        self.coef = np.linalg.solve(Z.T @ Z + pen, Z.T @ y)
        return self

    def predict(self, scenario: str) -> np.ndarray:
        """Forced GMST for 1850-2100 along the scenario [251]."""
        f = self.features(scenario)
        Z = np.concatenate([np.ones((len(f), 1)), f / self.scale], axis=1)
        return Z @ self.coef
