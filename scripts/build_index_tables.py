"""Index tables of every simulated training year, for emulators that reuse whole simulated years.

For each run, on the full grid (work/tables/<scenario>.npz):
    idx365   [Y, 15, C]  indices of the year over 365 days (Rx5day from windows inside the year only)
    idx364   [Y, 15, C]  the same over the first 364 days (the last year of every file is one day short)
    first4   [Y, 4, C]   precipitation (mm/day) of days 1-4, for the 5-day windows that straddle two years
    last4    [Y, 4, C]   precipitation of days 362-365
    last4_364[Y, 4, C]   precipitation of days 361-364
Only complete (365-day) years are tabulated: the truncated last year of each run is left out.

    python scripts/build_index_tables.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from climx.paths import WORK, SCENARIOS  # noqa: E402
from climx import data  # noqa: E402
from climx.indices import compute_indices  # noqa: E402

VARS = ("tasmax", "tasmin", "tas", "pr")
BLOCK = 12


def run(scenario: str, device: str) -> dict[str, np.ndarray]:
    thr = {k: torch.from_numpy(v).to(device) for k, v in data.load_thresholds().items()}
    arrs = {v: data.open_var(scenario, v) for v in VARS}
    n_full = arrs["pr"].shape[0] // 365
    res = {k: [] for k in ("idx365", "idx364", "first4", "last4", "last4_364")}
    for a in range(0, n_full, BLOCK):
        b = min(a + BLOCK, n_full)
        t = {v: torch.from_numpy(np.ascontiguousarray(arrs[v][a * 365:b * 365])).to(device)
             .view(b - a, 365, -1) for v in VARS}
        pr = t["pr"] * 86400
        res["idx365"].append(compute_indices(t["tasmax"], t["tasmin"], t["tas"], pr, thr).cpu())
        res["idx364"].append(compute_indices(t["tasmax"][:, :364], t["tasmin"][:, :364], t["tas"][:, :364],
                                             pr[:, :364], thr).cpu())
        res["first4"].append(pr[:, :4].cpu())
        res["last4"].append(pr[:, 361:365].cpu())
        res["last4_364"].append(pr[:, 360:364].cpu())
        del t, pr
    return {k: torch.cat(v).numpy() for k, v in res.items()}


def main() -> None:
    out_dir = WORK / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    for scen in SCENARIOS:
        out = out_dir / f"{scen}.npz"
        if out.exists():
            print("skip", scen)
            continue
        t0 = time.time()
        res = run(scen, "cuda")
        assert all(np.isfinite(v).all() for v in res.values())
        tmp = out.with_suffix(".tmp.npz")
        np.savez(tmp, **res)
        os.replace(tmp, out)
        print(f"{scen}: {res['idx365'].shape} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
