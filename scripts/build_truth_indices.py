"""The 15 annual indices of every training run on the full grid, computed on the GPU.

Output: work/indices/<scenario>.npy, float32 [n_years, 15, 192*288], index order as in
``climx.paths.INDEX_NAMES``. The last year of each run is one day short (364 days), as in the data
(and as in the hidden SSP2-4.5 targets, which the organisers computed from the same kind of file).

    python scripts/build_truth_indices.py
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
BLOCK = 24  # years per GPU batch


def run(scenario: str, device: str) -> np.ndarray:
    thr = {k: torch.from_numpy(v).to(device) for k, v in data.load_thresholds().items()}
    arrs = {v: data.open_var(scenario, v) for v in VARS}
    ny = data.n_years(scenario)
    n_full = arrs["pr"].shape[0] // 365          # complete years
    out = []
    prev4 = None
    for a in range(0, n_full, BLOCK):
        b = min(a + BLOCK, n_full)
        t = {v: torch.from_numpy(np.ascontiguousarray(arrs[v][a * 365:b * 365])).to(device)
             .view(b - a, 365, -1) for v in VARS}
        pr = t["pr"] * 86400
        first = prev4 if prev4 is not None else torch.full_like(pr[0, -4:], float("-inf"))
        p4 = torch.cat([first[None], pr[:-1, -4:]], dim=0)
        out.append(compute_indices(t["tasmax"], t["tasmin"], t["tas"], pr, thr, pr_prev4=p4).cpu())
        prev4 = pr[-1, -4:].clone()
        del t, pr, p4
    if ny > n_full:                              # truncated last year
        sl = slice(n_full * 365, arrs["pr"].shape[0])
        t = {v: torch.from_numpy(np.ascontiguousarray(arrs[v][sl])).to(device)[None] for v in VARS}
        pr = t["pr"] * 86400
        out.append(compute_indices(t["tasmax"], t["tasmin"], t["tas"], pr, thr, pr_prev4=prev4[None]).cpu())
    return torch.cat(out).numpy()


def main() -> None:
    device = "cuda"
    out_dir = WORK / "indices"
    out_dir.mkdir(parents=True, exist_ok=True)
    for scen in SCENARIOS:
        out = out_dir / f"{scen}.npy"
        if out.exists():
            print("skip", scen)
            continue
        t0 = time.time()
        res = run(scen, device)
        assert np.isfinite(res).all(), scen
        tmp = out.with_suffix(".tmp.npy")
        np.save(tmp, res)
        os.replace(tmp, out)
        print(f"{scen}: {res.shape} in {time.time() - t0:.0f}s, peak GPU "
              f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
