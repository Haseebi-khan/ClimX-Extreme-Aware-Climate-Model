"""Leave-one-scenario-out validation of emulator v0: resampling whole simulated years by warming level.

v0 produces, for each target year, N daily realisations that are unmodified simulated years drawn from the
other runs with a Gaussian kernel on the forced warming level; the 15 indices are computed on each
realisation and averaged without weights. Because the realisations are unmodified simulated years, the result
equals a weighted mean of tabulated per-year indices (scripts/build_index_tables.py), which makes this sweep
instant. The year 2100 is 364 days long, as in the data, so it uses the 364-day table.

    python scripts/validate_v0.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from climx.paths import WORK, INDEX_NAMES  # noqa: E402
from climx import data, analog  # noqa: E402
from climx.metric import Scorer  # noqa: E402

DEV = "cuda"
SSPS = analog.SSPS


def load_tables(land):
    tab = {}
    for r in analog.RUNS:
        z = np.load(WORK / "tables" / f"{r}.npz")
        tab[r] = {k: torch.from_numpy(z[k][:, :, land]).to(DEV) for k in ("idx365", "idx364")}
    return tab


def predict(weights: np.ndarray, pool: analog.Pool, tab) -> torch.Tensor:
    """weights [T=86, S] -> predicted indices [86, 15, P]; the last target year uses the 364-day table."""
    w = torch.from_numpy(weights).to(DEV, torch.float32)
    t365 = torch.cat([tab[r]["idx365"] for r in dict.fromkeys(pool.run)])
    t364 = torch.cat([tab[r]["idx364"] for r in dict.fromkeys(pool.run)])
    pred = torch.einsum("ts,sip->tip", w[:-1], t365)
    last = torch.einsum("ts,sip->tip", w[-1:], t364)
    return torch.cat([pred, last])


def main() -> None:
    land = data.land_index()
    scorer = Scorer(DEV)
    tab = load_tables(land)
    truth = {s: torch.from_numpy(np.load(WORK / "indices" / f"{s}.npy")[:, :, land]).to(DEV) for s in SSPS}
    gmst = analog.annual_gmst()

    results = {}
    print(f"{'config':58s} " + "  ".join(f"{s:>7s}" for s in SSPS) + "   mean126/370")
    for src_method in ("poly", "joint"):
        gwl = analog.source_gwl(gmst, src_method)
        for tgt in ("oracle", "emulated"):
            for h in (0.1, 0.15, 0.2, 0.3, 0.4, 0.6):
                for n_draws in (0, 200):
                    if n_draws and h not in (0.2, 0.3):
                        continue
                    scores, detail = [], {}
                    for held in SSPS:
                        runs = tuple(r for r in analog.RUNS if r != held)
                        pool = analog.build_pool(runs, gwl)
                        g_t = gwl[held] if tgt == "oracle" else analog.target_gwl(
                            held, gmst, tuple(s for s in SSPS if s != held))
                        w = analog.gaussian_weights(g_t, pool.gwl, h)
                        if n_draws:
                            w = analog.systematic_counts(w, n_draws) / n_draws
                        s, per_index, _ = scorer.score(truth[held], predict(w, pool, tab), detail=True)
                        scores.append(s)
                        detail[held] = per_index
                    label = f"source GWL {src_method:5s} | target GWL {tgt:8s} | h={h:<4} | draws {n_draws or 'exact'}"
                    results[label] = (scores, detail)
                    print(f"{label:58s} " + "  ".join(f"{s:+.4f}" for s in scores)
                          + f"   {np.mean(scores[:2]):+.4f}", flush=True)

    best = max(results, key=lambda k: np.mean(results[k][0][:2]) if "emulated" in k else -9)
    print("\nbest honest config (emulated target GWL):", best)
    scores, detail = results[best]
    print(f"{'index':8s}" + "".join(f"{s:>10s}" for s in SSPS))
    for k, name in enumerate(INDEX_NAMES):
        print(f"{name:8s}" + "".join(f"{detail[s][k]:10.4f}" for s in SSPS))


if __name__ == "__main__":
    main()
