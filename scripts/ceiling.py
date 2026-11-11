"""Where does a score sit on this metric? Ceiling and reference levels measured on the training runs.

Each SSP training run is treated in turn as the hidden test run. All predictors here act on the index maps
directly, which the rules forbid in a submission; they are used only to measure what is achievable.

A. Oracle fits to the run's OWN indices (they see the answer):
   polynomial in time or in the run's smoothed global-mean temperature (GMST), fitted per cell and index.
   The in-sample score is an upper bound on what a perfect forced-response model could reach; the
   "adjusted" score removes the part that comes from fitting noise (residual variance / (n - k)).
B. Honest cross-scenario predictors: the index of the held-out run is predicted from the OTHER runs only,
   by kernel regression on smoothed GMST (local-constant and local-linear, several bandwidths).
C. Reference points: the other runs' same-year values, and the held-out run's own temporal mean (= 0).

    python scripts/ceiling.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from climx.paths import WORK, INDEX_NAMES  # noqa: E402
from climx import data  # noqa: E402
from climx.metric import Scorer, VAR_EPS  # noqa: E402

SSPS = ("ssp126", "ssp370", "ssp585")
DEV = "cuda"


def smooth_gmst(g: np.ndarray, deg: int) -> np.ndarray:
    x = np.linspace(-1, 1, len(g))
    return np.polyval(np.polyfit(x, g, deg), x)


def hat_matrix(x: np.ndarray, deg: int) -> torch.Tensor:
    """Hat matrix of a polynomial regression of degree ``deg`` on the covariate x."""
    z = (x - x.mean()) / (x.std() + 1e-12)
    X = np.stack([z ** k for k in range(deg + 1)], axis=1)
    H = X @ np.linalg.solve(X.T @ X, X.T)
    return torch.from_numpy(H).to(DEV)


def kernel_smoother(g_t: np.ndarray, g_s: np.ndarray, h: float, local_linear: bool) -> torch.Tensor:
    """Smoother matrix L [T, S]: prediction at target GMST g_t from source years with GMST g_s."""
    d = g_s[None, :] - g_t[:, None]
    w = np.exp(-0.5 * (d / h) ** 2)
    if not local_linear:
        return torch.from_numpy(w / w.sum(1, keepdims=True)).to(DEV)
    s0, s1, s2 = w.sum(1), (w * d).sum(1), (w * d * d).sum(1)
    L = w * (s2[:, None] - d * s1[:, None]) / (s0 * s2 - s1 ** 2)[:, None]
    return torch.from_numpy(L).to(DEV)


def adjusted_score(scorer: Scorer, truth: torch.Tensor, pred: torch.Tensor, k: int):
    """Score after inflating the in-sample MSE by n / (n - k) (unbiased residual variance)."""
    n = truth.shape[0]
    mean = truth.mean(dim=0, keepdim=True)
    scaled = mean + (pred - mean)  # same prediction; the inflation is applied through the residuals
    resid = (truth - scaled) * np.sqrt(n / (n - k))
    return scorer.score(truth, truth - resid, detail=True)


def main() -> None:
    scorer = Scorer(DEV)
    land = data.land_index()
    ann = np.load(WORK / "annual.npz")
    idx = {s: torch.from_numpy(np.load(WORK / "indices" / f"{s}.npy")[:, :, land]).to(DEV, torch.float64)
           for s in ("historical",) + SSPS}
    gm_raw = {s: ann[f"{s}_gmst"] for s in idx}
    pre = gm_raw["historical"][:50].mean()
    gm = {s: smooth_gmst(gm_raw[s], 3) - pre for s in SSPS}
    gm["historical"] = smooth_gmst(gm_raw["historical"], 6) - pre
    print("smoothed GMST anomaly vs 1850-1899 (K):")
    for s in gm:
        print(f"   {s:10s} start {gm[s][0]:+.2f}  end {gm[s][-1]:+.2f}   interannual sd of residual "
              f"{np.std(gm_raw[s] - pre - gm[s]):.3f}")

    rows = []  # (label, {scenario: (S, per-index)})

    def add(label, fn):
        rows.append((label, {s: fn(s) for s in SSPS}))
        print(f"{label:46s} " + "  ".join(f"{rows[-1][1][s][0]:+.4f}" for s in SSPS), flush=True)

    print(f"\n{'predictor':46s} " + "  ".join(f"{s:>7s}" for s in SSPS))
    t_axis = np.arange(86, dtype=np.float64)
    # A. oracle fits on the run's own indices
    for label, cov, deg in [("A  own-run fit: linear in time", "t", 1), ("A  own-run fit: cubic in time", "t", 3),
                            ("A  own-run fit: linear in GMST", "g", 1), ("A  own-run fit: quadratic in GMST", "g", 2)]:
        def fit(s, cov=cov, deg=deg, adj=False):
            H = hat_matrix(t_axis if cov == "t" else gm[s], deg)
            pred = torch.einsum("ts,sip->tip", H, idx[s])
            if adj:
                return adjusted_score(scorer, idx[s], pred, deg + 1)[:2]
            return scorer.score(idx[s], pred, detail=True)[:2]
        add(label + " (in-sample)", fit)
        add(label + " (adjusted)", lambda s, f=fit: f(s, adj=True))
    # B. honest cross-scenario kernel regression on GMST
    for ll in (False, True):
        for h in (0.05, 0.1, 0.2, 0.4, 0.8):
            def kr(s, h=h, ll=ll, with_hist=True):
                src = [r for r in idx if r != s and (with_hist or r != "historical")]
                g_s = np.concatenate([gm[r] for r in src])
                y_s = torch.cat([idx[r] for r in src])
                L = kernel_smoother(gm[s], g_s, h, ll)
                return scorer.score(idx[s], torch.einsum("ts,sip->tip", L, y_s), detail=True)[:2]
            add(f"B  other runs, {'local-linear' if ll else 'local-const '} on GMST, h={h}", kr)
    # C. references
    def same_year(s):
        others = [r for r in SSPS if r != s]
        return scorer.score(idx[s], torch.stack([idx[r] for r in others]).mean(0), detail=True)[:2]
    add("C  mean of the other two SSPs, same year", same_year)

    print("\nPer-index scores (mean over the three held-out runs):")
    pick = [r for r in rows if "cubic in time (adjusted)" in r[0] or "quadratic in GMST (adjusted)" in r[0]
            or "quadratic in GMST (in-sample)" in r[0] or "local-linear on GMST, h=0.2" in r[0]
            or "local-const  on GMST, h=0.2" in r[0] or r[0].startswith("C ")]
    print(f"{'index':8s}" + "".join(f"{lab[3:30]:>30s}" for lab, _ in pick))
    for k, name in enumerate(INDEX_NAMES):
        print(f"{name:8s}" + "".join(f"{np.mean([res[s][1][k] for s in SSPS]):30.4f}" for _, res in pick))
    print(f"{'ALL':8s}" + "".join(f"{np.mean([res[s][0] for s in SSPS]):30.4f}" for _, res in pick))

    # share of scored cells that pass the variance test, and signal fraction per index
    print("\nScored cells per index (ssp370): " + ", ".join(
        f"{n} {int((idx['ssp370'][:, k].var(0, unbiased=False) > VAR_EPS).sum())}" for k, n in enumerate(INDEX_NAMES)))


if __name__ == "__main__":
    main()
