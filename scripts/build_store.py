"""Convert the full-resolution NetCDF files into flat float32 arrays for fast access.

Output: work/store/<scenario>/<var>.f32, a raw float32 array of shape [T, 192*288] (days x cells, cells in
latitude-major order), plus work/store/meta.json. The historical run is the concatenation of its two files.
Finished arrays are skipped, and each array is written under a temporary name and renamed when complete,
so the script can be stopped and started again.

    python scripts/build_store.py [--vars tasmax tasmin tas pr] [--workers 12]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from climx.paths import HF_FULL, WORK, NLAT, NLON  # noqa: E402

FILES = {
    "historical": ["NorESM2-MM_r1i1p1f1_with_forcing_hist_train_1850_1931.nc",
                   "NorESM2-MM_r1i1p1f1_with_forcing_hist_train_1932_2014.nc"],
    "ssp126": ["NorESM2-MM_r1i1p1f1_with_forcing_train_ssp126.nc"],
    "ssp370": ["NorESM2-MM_r1i1p1f1_with_forcing_train_ssp370.nc"],
    "ssp585": ["NorESM2-MM_r1i1p1f1_with_forcing_train_ssp585.nc"],
}
START_YEAR = {"historical": 1850, "ssp126": 2015, "ssp370": 2015, "ssp585": 2015}
BLOCK = 365


def n_days(scenario: str) -> int:
    import netCDF4
    total = 0
    for name in FILES[scenario]:
        with netCDF4.Dataset(HF_FULL / name) as ds:
            total += len(ds.dimensions["time"])
    return total


def convert(scenario: str, var: str) -> str:
    import netCDF4
    out_dir = WORK / "store" / scenario
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{var}.f32"
    if out.exists():
        return f"skip {scenario}/{var}"
    t0 = time.time()
    total = n_days(scenario)
    tmp = out.with_suffix(".f32.tmp")
    arr = np.memmap(tmp, dtype=np.float32, mode="w+", shape=(total, NLAT * NLON))
    pos = 0
    for name in FILES[scenario]:
        with netCDF4.Dataset(HF_FULL / name) as ds:
            v = ds[var]
            v.set_auto_mask(False)
            nt = v.shape[1]
            for a in range(0, nt, BLOCK):
                b = min(a + BLOCK, nt)
                block = np.asarray(v[0, a:b], dtype=np.float32).reshape(b - a, -1)
                if not np.isfinite(block).all():
                    raise ValueError(f"non-finite values in {name}:{var} days {a}-{b}")
                arr[pos:pos + (b - a)] = block
                pos += b - a
    assert pos == total
    arr.flush()
    del arr
    os.replace(tmp, out)
    return f"done {scenario}/{var} [{total} days] in {time.time() - t0:.0f}s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vars", nargs="+", default=["tasmax", "tasmin", "tas", "pr"])
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    import netCDF4
    with netCDF4.Dataset(HF_FULL / FILES["ssp126"][0]) as ds:
        lat = np.asarray(ds["lat"][:], dtype=np.float64)
        lon = np.asarray(ds["lon"][:], dtype=np.float64)
    meta = {"nlat": NLAT, "nlon": NLON, "lat": lat.tolist(), "lon": lon.tolist(),
            "start_year": START_YEAR, "calendar": "noleap",
            "n_days": {s: n_days(s) for s in FILES}, "dtype": "float32",
            "layout": "[day, cell] with cell = lat_index * nlon + lon_index"}
    (WORK / "store").mkdir(parents=True, exist_ok=True)
    (WORK / "store" / "meta.json").write_text(json.dumps(meta, indent=1))

    jobs = [(s, v) for s in FILES for v in args.vars]
    failed = False
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(convert, s, v): (s, v) for s, v in jobs}
        for f in as_completed(futs):
            try:
                print(time.strftime("%H:%M:%S"), f.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                failed = True
                print(time.strftime("%H:%M:%S"), "FAILED", futs[f], repr(e), flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
