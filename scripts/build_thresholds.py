"""Day-of-year percentile thresholds used by WSDI, CSDI and R95p.

They are computed with xclim's ``percentile_doy`` (5-day window) on the historical base period 1961-1990,
exactly as the official evaluation does (see src/climx/reference_xclim.py), and stored as
work/thresholds.npz:

    tx90 [365, 192*288]  90th percentile of tasmax (K)
    tn10 [365, 192*288]  10th percentile of tasmin (K)
    pr95 [365, 192*288]  95th percentile of pr, converted to mm/day

The values are kept in float64, the precision xclim returns and the official comparison uses.

    python scripts/build_thresholds.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from climx.paths import WORK, NLAT, NLON  # noqa: E402
from climx import data  # noqa: E402
from climx import reference_xclim as ref  # noqa: E402

BASE = ("1961-01-01", "1990-12-31")
SPEC = {"tx90": ("tasmax", 90.0), "tn10": ("tasmin", 10.0), "pr95": ("pr", 95.0)}
CHUNK = 24  # latitude rows per call (memory)


def base_dataset(var: str, rows: slice) -> xr.Dataset:
    """Historical 1961-1990 daily values of one variable for some latitude rows, as an xarray Dataset with a
    noleap time axis (what ``dataset.hist`` provides in the official pipeline)."""
    y0, y1 = 1961, 1990
    a = data.year_slice("historical", y0).start
    b = data.year_slice("historical", y1).stop
    arr = data.open_var("historical", var)
    nrow = rows.stop - rows.start
    block = np.asarray(arr[a:b, rows.start * NLON:rows.stop * NLON]).reshape(b - a, nrow, NLON)
    time_ = xr.date_range(f"{y0}-01-01", periods=b - a, freq="D", calendar="noleap", use_cftime=True)
    units = {"tasmax": "K", "tasmin": "K", "pr": "kg m-2 s-1"}[var]
    da = xr.DataArray(block, dims=("time", "lat", "lon"),
                      coords={"time": time_, "lat": np.arange(rows.start, rows.stop), "lon": np.arange(NLON)},
                      attrs={"units": units})
    return xr.Dataset({var: da})


def main() -> None:
    out = WORK / "thresholds.npz"
    if out.exists():
        print("exists:", out)
        return
    res = {}
    for key, (var, q) in SPEC.items():
        t0 = time.time()
        full = np.empty((365, NLAT, NLON), dtype=np.float64)
        for r0 in range(0, NLAT, CHUNK):
            rows = slice(r0, min(r0 + CHUNK, NLAT))
            ds = base_dataset(var, rows)
            thr = ref.percentile_threshold(ds[var], q, BASE, 5)
            if key == "pr95":
                thr = ref.precip_mm_day(thr)
            thr = thr.transpose("dayofyear", "lat", "lon")
            assert thr.shape[0] == 365 and list(thr["dayofyear"].values) == list(range(1, 366))
            assert thr.dtype == np.float64
            full[:, rows] = thr.values
        assert np.isfinite(full).all()
        res[key] = full.reshape(365, NLAT * NLON)
        print(f"{key}: {time.time() - t0:.0f}s  min {full.min():.4f} max {full.max():.4f}", flush=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, **res)
    os.replace(tmp, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
