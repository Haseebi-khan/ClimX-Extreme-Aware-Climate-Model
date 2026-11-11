"""Annual means of the daily variables per cell, and the global-mean temperature of every run.

Output: work/annual.npz with, for each run,
    <scenario>_<var>  float32 [n_years, 192*288]   annual mean (pr in mm/day)
    <scenario>_gmst   float64 [n_years]            area-weighted global-mean tas (K)

    python scripts/build_annual_means.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from climx.paths import WORK, SCENARIOS, NLON  # noqa: E402
from climx import data  # noqa: E402

VARS = ("tas", "tasmax", "tasmin", "pr")


def main() -> None:
    out = WORK / "annual.npz"
    if out.exists():
        print("exists:", out)
        return
    dev = "cuda"
    w = np.repeat(np.cos(np.deg2rad(data.lats())), NLON)
    w = torch.from_numpy(w / w.sum()).to(dev)
    res = {}
    for scen in SCENARIOS:
        ny = data.n_years(scen)
        for var in VARS:
            arr = data.open_var(scen, var)
            means = []
            for i in range(ny):
                sl = slice(i * 365, min((i + 1) * 365, arr.shape[0]))
                x = torch.from_numpy(np.ascontiguousarray(arr[sl])).to(dev)
                m = x.double().mean(dim=0)
                means.append(m * 86400 if var == "pr" else m)
            m = torch.stack(means)
            res[f"{scen}_{var}"] = m.float().cpu().numpy()
            if var == "tas":
                res[f"{scen}_gmst"] = (m * w).sum(dim=1).cpu().numpy()
        g = res[f"{scen}_gmst"]
        print(f"{scen}: GMST first {g[0]:.3f} last {g[-1]:.3f} K", flush=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, **res)
    os.replace(tmp, out)


if __name__ == "__main__":
    main()
