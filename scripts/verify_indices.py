"""Check the GPU index code (src/climx/indices.py) against the xclim reference on real data.

For a random sample of grid cells the 15 annual indices of a whole run are computed twice: with the xclim
reference (src/climx/reference_xclim.py, a call-for-call transcription of the organisers' index code) and
with the GPU implementation. The two must agree.

    python scripts/verify_indices.py [--cells 1500] [--scenarios ssp370 historical]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from climx.paths import INDEX_NAMES  # noqa: E402
from climx import data  # noqa: E402
from climx import reference_xclim as ref  # noqa: E402
from climx.indices import compute_indices  # noqa: E402

UNITS = {"tasmax": "K", "tasmin": "K", "tas": "K", "pr": "kg m-2 s-1"}


def as_dataset(scenario: str, cells: np.ndarray) -> xr.Dataset:
    y0 = data.start_year(scenario)
    out = {}
    n = None
    for var in UNITS:
        arr = np.asarray(data.open_var(scenario, var)[:, cells])
        n = arr.shape[0]
        out[var] = (("time", "lat", "lon"), arr[:, None, :], {"units": UNITS[var]})
    time_ = xr.date_range(f"{y0}-01-01", periods=n, freq="D", calendar="noleap", use_cftime=True)
    return xr.Dataset(out, coords={"time": time_, "lat": [0], "lon": np.arange(len(cells))})


def official_indices(ds: xr.Dataset, hist: xr.Dataset) -> dict[str, np.ndarray]:
    thr = {k: ref.percentile_threshold(hist[v], q).compute()
           for k, (v, q) in {"tx90": ("tasmax", 90.0), "tn10": ("tasmin", 10.0), "pr95": ("pr", 95.0)}.items()}
    res = ref.compute_reference(ds["tasmax"], ds["tasmin"], ds["tas"], ds["pr"],
                                thr["tx90"], thr["tn10"], thr["pr95"])
    return {name: np.asarray(da.transpose("year", "lat", "lon").values[:, 0, :], dtype=np.float64)
            for name, da in res.items()}


def gpu_indices(scenario: str, cells: np.ndarray, device: str) -> np.ndarray:
    thr_np = data.load_thresholds()
    thr = {k: torch.from_numpy(v[:, cells]).to(device) for k, v in thr_np.items()}
    arrs = {v: data.open_var(scenario, v) for v in UNITS}
    y0 = data.start_year(scenario)
    out = []
    prev4 = None
    for i in range(data.n_years(scenario)):
        sl = data.year_slice(scenario, y0 + i)
        t = {v: torch.from_numpy(np.asarray(arrs[v][sl][:, cells])).to(device) for v in UNITS}
        pr = t["pr"] * 86400
        out.append(compute_indices(t["tasmax"], t["tasmin"], t["tas"], pr, thr, pr_prev4=prev4).cpu().numpy())
        prev4 = pr[-4:]
    return np.stack(out)  # [year, index, cell]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=1500)
    ap.add_argument("--scenarios", nargs="+", default=["ssp370", "historical"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rng = np.random.default_rng(args.seed)
    land = data.land_index()
    ocean = np.setdiff1d(np.arange(192 * 288), land)
    cells = np.sort(np.concatenate([rng.choice(land, args.cells, replace=False),
                                    rng.choice(ocean, max(args.cells // 5, 1), replace=False)]))
    print(f"{len(cells)} cells, device {device}", flush=True)
    hist = as_dataset("historical", cells)
    ok = True
    for scen in args.scenarios:
        ds = hist if scen == "historical" else as_dataset(scen, cells)
        t0 = time.time()
        ref = official_indices(ds, hist)
        t1 = time.time()
        mine = gpu_indices(scen, cells, device)
        t2 = time.time()
        print(f"== {scen}: xclim reference {t1 - t0:.0f}s, gpu {t2 - t1:.1f}s, years {mine.shape[0]}", flush=True)
        for k, name in enumerate(INDEX_NAMES):
            r, m = ref[name], mine[:, k, :].astype(np.float64)
            if r.shape != m.shape:
                print(f"   {name:7s} SHAPE {r.shape} vs {m.shape}")
                ok = False
                continue
            diff = np.abs(r - m)
            tol = 1e-3 * np.maximum(1.0, np.abs(r))
            bad = diff > tol
            print(f"   {name:7s} max|diff| {np.nanmax(diff):10.4g}   mismatching {bad.sum():7d} of {bad.size}"
                  f"   mean ref {np.nanmean(r):9.3f}", flush=True)
            if bad.any():
                ok = False
                yy, cc = np.nonzero(bad)
                for j in range(min(3, len(yy))):
                    print(f"        year {yy[j]} cell {cells[cc[j]]}: official {r[yy[j], cc[j]]:.5f} gpu {m[yy[j], cc[j]]:.5f}")
    print("ALL MATCH" if ok else "MISMATCHES FOUND")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
