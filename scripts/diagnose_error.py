"""Where does the remaining error of an emulator come from? For a held-out run, the prediction error of each
cell and index is split into a constant offset, a linear trend, curvature (quadratic + cubic) and the rest,
all in units of the truth's temporal variance (so the columns add up to 1 - R2).

    python scripts/diagnose_error.py --holdout ssp370
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from climx.paths import WORK, INDEX_NAMES  # noqa: E402
from climx import data, analog, engine  # noqa: E402
from climx.metric import Scorer, VAR_EPS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="ssp370")
    ap.add_argument("--h", type=float, default=0.5)
    ap.add_argument("--stride", type=int, default=4)
    args = ap.parse_args()
    dev = "cuda"
    cells = data.land_index()[::args.stride]
    scorer = Scorer(dev, cells)
    gmst = analog.annual_gmst()
    gwl = analog.source_gwl(gmst, "joint")
    train = tuple(s for s in analog.SSPS if s != args.holdout)
    pool = analog.build_pool(("historical",) + train, gwl)
    dp = engine.DailyPool(pool, cells, dev)
    truth = torch.from_numpy(np.load(WORK / "indices" / f"{args.holdout}.npy")[:, :, cells]).to(dev, torch.float64)
    g_t = analog.target_gwl(args.holdout, gmst, train)
    cm = engine.fit_climate(dp)
    counts = analog.systematic_counts(analog.gaussian_weights(g_t, pool.gwl, args.h), 200)
    pred = engine.emulate(dp, cm, engine.Transform(shift=True), g_t, counts).double()
    s, per_index, _ = scorer.score(truth, pred, detail=True)
    print(f"held out {args.holdout}, shift h={args.h}: score {s:+.4f}")

    err = pred - truth                                           # [T, I, P]
    t = np.linspace(-1, 1, err.shape[0])
    # orthonormal polynomial basis up to degree 3
    q, _ = np.linalg.qr(np.stack([t ** k for k in range(4)], axis=1))
    q = torch.from_numpy(q).to(dev)
    proj = torch.einsum("tk,tip->kip", q, err) ** 2 / err.shape[0]      # variance captured by each component
    var = truth.var(dim=0, unbiased=False)
    valid = var > VAR_EPS
    w = scorer.cosw[None] * valid
    def avg(x):                                                  # area-weighted mean over valid cells
        return ((torch.where(valid, x / var.clamp(min=VAR_EPS), torch.zeros_like(x)) * w).sum(1) / w.sum(1)).cpu().numpy()
    mse = (err ** 2).mean(0)
    comp = {"offset": avg(proj[0]), "trend": avg(proj[1]), "curvature": avg(proj[2] + proj[3]),
            "rest": avg(mse - proj.sum(0)), "total": avg(mse)}
    n = err.shape[0]
    print("error variance / truth variance, area-weighted over scored cells (white noise would put 1/86 = 0.012 of "
          "the total in each of offset and trend, 0.023 in curvature)")
    print(f"{'index':8s}{'1-R2':>8s}{'offset':>9s}{'trend':>9s}{'curv.':>9s}{'rest':>9s}   excess over white noise")
    tot_excess = 0
    for k, name in enumerate(INDEX_NAMES):
        rest_rate = comp["rest"][k] / (n - 4)
        excess = comp["offset"][k] + comp["trend"][k] + comp["curvature"][k] - 4 * rest_rate
        tot_excess += excess
        print(f"{name:8s}{comp['total'][k]:8.3f}{comp['offset'][k]:9.4f}{comp['trend'][k]:9.4f}"
              f"{comp['curvature'][k]:9.4f}{comp['rest'][k]:9.3f}   {excess:+.4f} "
              f"(offset {comp['offset'][k] - rest_rate:+.4f}, trend {comp['trend'][k] - rest_rate:+.4f})")
    print(f"mean excess over the 15 indices: {tot_excess / 15:+.4f} of the truth variance")


if __name__ == "__main__":
    main()
