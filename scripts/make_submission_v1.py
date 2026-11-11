"""Emulator v1 on the full grid: a submission file for SSP2-4.5, or an end-to-end check on a held-out run.

Every realisation of a target year is a simulated training year (drawn with a Gaussian kernel on the forced
warming level) whose daily temperatures are shifted to the target warming level with the per-cell seasonal
climate model of ``climx.engine``. The 15 indices are computed on every realisation and averaged without
weights. The grid is processed in chunks of cells to fit the GPU.

    python scripts/make_submission_v1.py --h 0.5 --draws 200 --out submissions/v1_shift_h0.5_n200.parquet
    python scripts/make_submission_v1.py --h 0.5 --draws 200 --holdout ssp370    # scored with the official code
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from climx.paths import WORK, INDEX_NAMES, NLAT, NLON  # noqa: E402
from climx import analog, engine, submission, official_rnNSE  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, default=0.5, help="kernel width on the warming level (K)")
    ap.add_argument("--draws", type=int, default=200, help="realisations per target year")
    ap.add_argument("--scale", action="store_true", help="also rescale the temperature anomalies")
    ap.add_argument("--pr", action="store_true", help="also scale precipitation by the ratio of seasonal means")
    ap.add_argument("--chunks", type=int, default=4, help="number of cell chunks")
    ap.add_argument("--holdout", default=None, help="training SSP to treat as the test scenario")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = "cuda"

    gmst = analog.annual_gmst()
    gwl = analog.source_gwl(gmst, "joint")
    target = args.holdout or "ssp245"
    train = tuple(s for s in analog.SSPS if s != args.holdout)
    pool = analog.build_pool(("historical",) + train, gwl)
    g_t = analog.target_gwl(target, gmst, train)
    counts = analog.systematic_counts(analog.gaussian_weights(g_t, pool.gwl, args.h), args.draws)
    tr = engine.Transform(shift=True, scale=args.scale, pr=args.pr)
    print(f"target {target}: warming level {g_t[0]:.2f} -> {g_t[-1]:.2f} K; pool {len(pool)} years; "
          f"distinct source years per target year: median {int(np.median((counts > 0).sum(1)))}", flush=True)

    n_cells = NLAT * NLON
    pred = np.empty((86, len(INDEX_NAMES), n_cells), dtype=np.float32)
    bounds = np.linspace(0, n_cells, args.chunks + 1).astype(int)
    for a, b in zip(bounds[:-1], bounds[1:]):
        t0 = time.time()
        cells = np.arange(a, b)
        dp = engine.DailyPool(pool, cells, dev)
        cm = engine.fit_climate(dp)
        pred[:, :, a:b] = engine.emulate(dp, cm, tr, g_t, counts).cpu().numpy()
        print(f"   cells {a}-{b}: {time.time() - t0:.0f}s, peak GPU {torch.cuda.max_memory_allocated() / 2**30:.0f} GB",
              flush=True)
        del dp, cm
        torch.cuda.empty_cache()
    assert np.isfinite(pred).all()

    if args.holdout:
        truth = np.load(WORK / "indices" / f"{args.holdout}.npy")
        score = official_rnNSE.score(submission.to_frame(truth), submission.to_frame(pred), "row_id")
        print(f"official rnNSE score on held-out {args.holdout}: {score:+.5f}")
        return

    name = f"v1_shift{'_scale' if args.scale else ''}{'_pr' if args.pr else ''}_h{args.h}_n{args.draws}"
    path = submission.write(pred, ROOT / (args.out or f"submissions/{name}.parquet"))
    meta = {"emulator": "v1 analog years shifted to the target warming level", "h": args.h, "draws": args.draws,
            "transform": {"shift": True, "scale": args.scale, "pr": args.pr},
            "target_gwl_2015_2100": [float(g_t[0]), float(g_t[-1])], "pool_years": len(pool),
            "column_means": {n: float(pred[:, k].mean()) for k, n in enumerate(INDEX_NAMES)}}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print("wrote", path, f"({path.stat().st_size / 2**20:.0f} MB)")


if __name__ == "__main__":
    main()
