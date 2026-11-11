"""The array scorer must return the same number as the organisers' rnNSE.score on full submissions.

    python tests/test_metric.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from climx import data, official_rnNSE  # noqa: E402
from climx.metric import Scorer  # noqa: E402
from climx.paths import WORK, INDEX_NAMES  # noqa: E402


def frame(arr: np.ndarray) -> pd.DataFrame:
    """[T, 15, cells] -> submission-style frame (time-major, then latitude, then longitude)."""
    t, n, c = arr.shape
    df = pd.DataFrame(arr.transpose(0, 2, 1).reshape(t * c, n).astype(np.float64), columns=list(INDEX_NAMES))
    df.insert(0, "row_id", np.arange(len(df), dtype=np.int64))
    return df


def main() -> None:
    truth = np.load(WORK / "indices" / "ssp126.npy")
    other = np.load(WORK / "indices" / "ssp370.npy")
    land = data.land_index()
    scorer = Scorer()
    rng = np.random.default_rng(0)
    cases = {
        "another scenario": other,
        "temporal mean of the truth": np.broadcast_to(truth.mean(0, keepdims=True), truth.shape).copy(),
        "truth plus noise": truth + rng.standard_normal(truth.shape).astype(np.float32) * truth.std(0, keepdims=True),
    }
    ok = True
    for name, pred in cases.items():
        ref = official_rnNSE.score(frame(truth), frame(pred), "row_id")
        mine = scorer.score(torch.from_numpy(truth[:, :, land]), torch.from_numpy(pred[:, :, land]))
        good = abs(ref - mine) < 1e-9
        ok &= good
        print(f"{name:28s} official {ref:+.10f}  array scorer {mine:+.10f}  {'OK' if good else 'DIFFERENT'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
