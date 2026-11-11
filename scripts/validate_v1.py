"""Leave-one-scenario-out validation of emulator v1 (simulated years moved to the target warming level).

Each configuration is run with every training SSP held out in turn. ``--stride k`` uses every k-th scored
cell (faster sweeps; region and index averages are kept), ``--stride 1`` all of them.

    python scripts/validate_v1.py --stride 4 --configs sweep1
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
from climx.paths import WORK, INDEX_NAMES  # noqa: E402
from climx import data, analog, engine  # noqa: E402
from climx.metric import Scorer  # noqa: E402

DEV = "cuda"
SSPS = analog.SSPS

# name -> dict(h=kernel width, tr=Transform or None, feat=state features, nh/nhv=harmonics, ridge, gwl)
T = engine.Transform


def cfg(h, tr=None, feat=("g",), nh=4, nhv=2, ridge=0.0, gwl="emulated", jv=1, draws=None, min_year=0):
    """min_year: source years before it get no weight (e.g. 1950 drops the early historical period)."""
    return dict(h=h, tr=tr, feat=tuple(feat), nh=nh, nhv=nhv, ridge=ridge, gwl=gwl, jv=jv, draws=draws,
                min_year=min_year)


SH = T(shift=True)
SC = T(shift=True, scale=True)
SWEEPS = {
    "sweep3": {
        "shift h=0.4": cfg(0.4, SH),
        "shift h=0.5": cfg(0.5, SH),
        "shift h=0.6": cfg(0.6, SH),
        "shift h=0.7": cfg(0.7, SH),
        "shift+scale h=0.6": cfg(0.6, SC),
        "shift+scale h=0.7": cfg(0.7, SC),
        "shift h=0.5, 800 draws": cfg(0.5, SH, draws=800),
        "shift h=0.5, sources from 1950": cfg(0.5, SH, min_year=1950),
        "shift h=0.5, sources from 2015": cfg(0.5, SH, min_year=2015),
    },
    "check": {"v0 h=0.3 (engine, no adjustment)": cfg(0.3)},
    "sweep2": {
        "shift g            h=0.5": cfg(0.5, SH),
        "shift g, oracle GWL h=0.5": cfg(0.5, SH, gwl="oracle"),
        "shift g,g2         h=0.5": cfg(0.5, SH, ("g", "g2")),
        "shift g,so2        h=0.5": cfg(0.5, SH, ("g", "so2")),
        "shift g,so2,co2    h=0.5": cfg(0.5, SH, ("g", "so2", "co2")),
        "shift g,g2,so2     h=0.5": cfg(0.5, SH, ("g", "g2", "so2")),
        "shift g,so2,bc     h=0.5": cfg(0.5, SH, ("g", "so2", "bc")),
        "shift g,so2,co2,ch4 h=0.5": cfg(0.5, SH, ("g", "so2", "co2", "ch4")),
        "shift g,so2        h=0.8": cfg(0.8, SH, ("g", "so2")),
        "shift g,so2,co2    h=0.8": cfg(0.8, SH, ("g", "so2", "co2")),
        "shift g,so2,co2    h=1.5": cfg(1.5, SH, ("g", "so2", "co2")),
        "shift g  nh=2      h=0.5": cfg(0.5, SH, nh=2),
        "shift g  nh=6      h=0.5": cfg(0.5, SH, nh=6),
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--configs", default="sweep2")
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--holdouts", nargs="+", default=list(SSPS))
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    cells = data.land_index()[::args.stride]
    scorer = Scorer(DEV, cells)
    gmst = analog.annual_gmst()
    gwl = analog.source_gwl(gmst, "joint")
    configs = SWEEPS[args.configs]
    results = {name: {} for name in configs}
    log = WORK / "validation" / f"{args.tag or args.configs}_stride{args.stride}.json"
    log.parent.mkdir(parents=True, exist_ok=True)

    for held in args.holdouts:
        t0 = time.time()
        train = tuple(s for s in SSPS if s != held)
        pool = analog.build_pool(("historical",) + train, gwl)
        dp = engine.DailyPool(pool, cells, DEV)
        truth = torch.from_numpy(np.load(WORK / "indices" / f"{held}.npy")[:, :, cells]).to(DEV)
        g_emul = analog.target_gwl(held, gmst, train)
        years = np.arange(2015, 2101)
        print(f"== held out {held}: pool {len(pool)} years, {len(cells)} cells, loaded in {time.time() - t0:.0f}s, "
              f"GPU {torch.cuda.memory_allocated() / 2**30:.0f} GB", flush=True)
        models = {}
        for name, c in configs.items():
            t1 = time.time()
            g_t = gwl[held] if c["gwl"] == "oracle" else g_emul
            dp.state = torch.from_numpy(analog.state_matrix(pool.run, pool.year, pool.gwl, c["feat"])).to(DEV)
            t_state = analog.state_matrix(np.full(86, held), years, g_t, c["feat"])
            cm = None
            if c["tr"] is not None:
                key = (c["feat"], c["nh"], c["nhv"], c["ridge"], c["jv"])
                if key not in models:
                    models[key] = engine.fit_climate(dp, c["nh"], c["nhv"], c["jv"], ridge=c["ridge"])
                cm = models[key]
            w = analog.gaussian_weights(g_t, pool.gwl, c["h"]) * (pool.year >= c["min_year"])[None]
            counts = analog.systematic_counts(w / w.sum(1, keepdims=True), c["draws"] or args.draws)
            pred = engine.emulate(dp, cm, c["tr"], t_state, counts)
            s, per_index, _ = scorer.score(truth, pred, detail=True)
            results[name][held] = {"score": s, "per_index": per_index.tolist()}
            print(f"   {name:28s} {s:+.4f}   ({time.time() - t1:.0f}s)", flush=True)
            log.write_text(json.dumps(results, indent=1))
        del dp, truth
        torch.cuda.empty_cache()

    print(f"\n{'config':30s}" + "".join(f"{s:>9s}" for s in args.holdouts) + "   mean  mean126/370")
    for name, res in results.items():
        vals = [res[s]["score"] for s in args.holdouts if s in res]
        m2 = np.mean([res[s]["score"] for s in ("ssp126", "ssp370") if s in res])
        print(f"{name:30s}" + "".join(f"{v:+9.4f}" for v in vals) + f"   {np.mean(vals):+.4f}   {m2:+.4f}")
    print("\nper index, mean over ssp126 and ssp370:")
    print(f"{'config':30s}" + "".join(f"{n:>7s}" for n in INDEX_NAMES))
    for name, res in results.items():
        pi = np.mean([res[s]["per_index"] for s in ("ssp126", "ssp370") if s in res], axis=0)
        print(f"{name:30s}" + "".join(f"{v:+7.3f}" for v in pi))


if __name__ == "__main__":
    main()
