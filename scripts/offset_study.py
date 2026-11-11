"""Can the scenario-dependent offset in local temperature be predicted from the aerosol emission maps?

For each held-out SSP, the annual-mean temperature of every scored cell is predicted from the other runs with
the v1 recipe (years drawn by warming level and shifted to the target state) under different shift models:

    g          local warming per kelvin of global warming
    g + A      plus a term in the regional aerosol emissions A_p (SO2 map smoothed over L km, linear or log)
    g + S      plus a term in global log SO2

Reported per held-out run, as area-weighted means over scored cells and in units of the interannual variance:
the squared 86-year mean error (the offset) and the mean squared error of the 86 annual values.

    python scripts/offset_study.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from climx.paths import WORK, HF_FULL, NLAT, NLON  # noqa: E402
from climx import data, analog  # noqa: E402
from climx.forcing import HIST_FILE, ALL_FILE  # noqa: E402

DEV = "cuda"
AER = {"so2": ("SO2",), "bc": ("BC_AX", "BC_N"), "om": ("OM_NI",)}


def annual_aerosol_maps() -> dict[str, np.ndarray]:
    """<run>_<agent> [n_years, 192*288] annual-mean emission maps; cached in work/aerosol_annual.npz."""
    path = WORK / "aerosol_annual.npz"
    if not path.exists():
        import netCDF4
        out = {}

        def add(ds, k, run):
            for name, vs in AER.items():
                x = sum(np.asarray(ds[v][k], dtype=np.float32) for v in vs)
                out[f"{run}_{name}"] = x.reshape(-1, 12, NLAT * NLON).mean(axis=1)
        with netCDF4.Dataset(HF_FULL / HIST_FILE) as ds:
            add(ds, 0, "historical")
        with netCDF4.Dataset(HF_FULL / ALL_FILE) as ds:
            for k, s in enumerate(str(x) for x in ds["forcing_scenario"][:]):
                add(ds, k, s)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, **out)
        os.replace(tmp, path)
    z = np.load(path)
    return {k: z[k] for k in z.files}


def smoothing_matrix(cells: np.ndarray, length_km: float) -> torch.Tensor:
    """Row-normalised Gaussian weights by great-circle distance, area-weighted: [len(cells), 192*288]."""
    lat = np.deg2rad(np.repeat(data.lats(), NLON))
    lon = np.deg2rad(np.tile(np.asarray(data.meta()["lon"]), NLAT))
    xyz = torch.from_numpy(np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], 1)).to(DEV)
    area = torch.from_numpy(np.cos(lat)).to(DEV)
    rows = []
    for a in range(0, len(cells), 2048):
        c = torch.from_numpy(cells[a:a + 2048]).to(DEV)
        d = torch.arccos((xyz[c] @ xyz.T).clamp(-1, 1)) * 6371.0
        w = torch.exp(-0.5 * (d / length_km) ** 2) * area[None]
        rows.append((w / w.sum(1, keepdim=True)).float())
    return torch.cat(rows)


def main() -> None:
    cells = data.land_index()
    ann = np.load(WORK / "annual.npz")
    aer = annual_aerosol_maps()
    gmst = analog.annual_gmst()
    gwl = analog.source_gwl(gmst, "joint")
    cosw = torch.from_numpy(np.cos(np.deg2rad(data.lats()))[cells // NLON]).to(DEV)
    years = np.arange(2015, 2101)
    tann = {r: torch.from_numpy(ann[f"{r}_tas"][:, cells]).to(DEV, torch.float64) for r in analog.RUNS}

    def regional(length, agent, log):
        W = smoothing_matrix(cells, length)
        out = {}
        ref = None
        for r in analog.RUNS:
            e = torch.from_numpy(aer[f"{r}_{agent}"]).to(DEV)
            s = (e @ W.T).double()                           # [Y, P]
            if r == "historical":
                ref = s[:20].mean(0).mean()                  # global scale: 1850-1869 mean
            out[r] = s
        for r in out:
            out[r] = torch.log(out[r] / ref + 0.1) if log else out[r] / ref / 20.0
        return out

    covs = {"g": None}
    for length in (700, 1500, 3000):
        covs[f"g + A(so2 lin {length}km)"] = regional(length, "so2", False)
        covs[f"g + A(so2 log {length}km)"] = regional(length, "so2", True)
    covs["g + A(bc log 1500km)"] = regional(1500, "bc", True)
    glob = {r: torch.from_numpy(analog.forcing_feature(r, "so2")).to(DEV)[:, None].expand(-1, len(cells)).double()
            for r in analog.RUNS}
    covs["g + S(global log so2)"] = glob

    print(f"{'shift model':28s}" + "".join(f"{s + ' offset':>16s}{'mse':>8s}" for s in analog.SSPS))
    for name, A in covs.items():
        line = f"{name:28s}"
        for held in analog.SSPS:
            train = tuple(s for s in analog.SSPS if s != held)
            pool = analog.build_pool(("historical",) + train, gwl)
            g_t = analog.target_gwl(held, gmst, train)
            w = torch.from_numpy(analog.gaussian_weights(g_t, pool.gwl, 0.5)).to(DEV)          # [T, S]
            y = torch.cat([tann[r][:len(gwl[r]) - 1] for r in ("historical",) + train])          # [S, P]
            g_s = torch.from_numpy(pool.gwl).to(DEV)
            cols = [torch.ones_like(y), g_s[:, None].expand_as(y)]
            if A is not None:
                cols.append(torch.cat([A[r][:len(gwl[r]) - 1] for r in ("historical",) + train]))
            X = torch.stack(cols, dim=2)                                                          # [S, P, K]
            xtx = torch.einsum("spk,spl->pkl", X, X)
            xty = torch.einsum("spk,sp->pk", X, y)
            beta = torch.linalg.solve(xtx, xty)                                                   # [P, K]
            gt = torch.from_numpy(g_t).to(DEV)
            # prediction: kernel mean of source values shifted to the target state
            pred = w @ y + beta[:, 1][None] * (gt[:, None] - (w @ g_s)[:, None])
            if A is not None:
                a_s = X[:, :, 2]
                pred = pred + beta[:, 2][None] * (A[held] - w @ a_s)
            truth = tann[held]
            t = torch.linspace(-1, 1, 86, device=DEV, dtype=torch.float64)
            Q = torch.linalg.qr(torch.stack([t ** k for k in range(4)], 1)).Q
            resid = truth - Q @ (Q.T @ truth)
            sig2 = (resid ** 2).sum(0) / (86 - 4)
            err = pred - truth
            off = (err.mean(0) ** 2 / sig2 * cosw).sum() / cosw.sum()
            mse = ((err ** 2).mean(0) / sig2 * cosw).sum() / cosw.sum()
            line += f"{float(off):16.4f}{float(mse):8.3f}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
