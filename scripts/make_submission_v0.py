"""Emulator v0 on the full grid: a submission file for SSP2-4.5, or an end-to-end check on a held-out run.

    python scripts/make_submission_v0.py --h 0.3 --draws 200 --out submissions/v0_h0.3_n200.parquet
    python scripts/make_submission_v0.py --h 0.3 --draws 200 --holdout ssp370      # scores with the official code
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from climx.paths import WORK, INDEX_NAMES  # noqa: E402
from climx import analog, resample, submission, official_rnNSE  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, default=0.3, help="kernel width on the warming level (K)")
    ap.add_argument("--draws", type=int, default=200, help="realisations per target year")
    ap.add_argument("--gwl", default="joint", choices=["joint", "poly"])
    ap.add_argument("--holdout", default=None, help="training SSP to treat as the test scenario")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gmst = analog.annual_gmst()
    gwl = analog.source_gwl(gmst, args.gwl)
    target = args.holdout or "ssp245"
    train_ssps = tuple(s for s in analog.SSPS if s != args.holdout)
    pool = analog.build_pool(("historical",) + train_ssps, gwl)
    g_t = analog.target_gwl(target, gmst, train_ssps)
    print(f"target {target}: emulated warming level {g_t[0]:.2f} K (2015) -> {g_t[-1]:.2f} K (2100); "
          f"pool of {len(pool)} simulated years from {('historical',) + train_ssps}")

    w = analog.gaussian_weights(g_t, pool.gwl, args.h)
    counts = analog.systematic_counts(w, args.draws)
    seq = resample.sequences_from_counts(counts)
    n_eff = 1.0 / ((counts / args.draws) ** 2).sum(axis=1)
    print(f"distinct source years per target year: min {int((counts > 0).sum(1).min())}, "
          f"median {int(np.median((counts > 0).sum(1)))}; effective sample size median {np.median(n_eff):.0f}")

    tab = resample.load_tables(pool, None)
    pred = resample.emulate(seq, tab).cpu().numpy()
    assert np.isfinite(pred).all()

    if args.holdout:
        truth = np.load(WORK / "indices" / f"{args.holdout}.npy")
        score = official_rnNSE.score(submission.to_frame(truth), submission.to_frame(pred), "row_id")
        print(f"official rnNSE score on held-out {args.holdout}: {score:+.5f}")
        return

    out = Path(args.out or f"submissions/v0_h{args.h}_n{args.draws}.parquet")
    path = submission.write(pred, ROOT / out)
    meta = {"emulator": "v0 analog-year resampling", "h": args.h, "draws": args.draws, "gwl": args.gwl,
            "target_gwl_2015_2100": [float(g_t[0]), float(g_t[-1])], "pool_years": len(pool),
            "column_means": {n: float(pred[:, k].mean()) for k, n in enumerate(INDEX_NAMES)}}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print("wrote", path, f"({path.stat().st_size / 2**20:.0f} MB)")


if __name__ == "__main__":
    main()
