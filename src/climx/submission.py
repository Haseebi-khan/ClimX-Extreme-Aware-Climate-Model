"""Writing and checking submission files."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .paths import INDEX_NAMES, NLAT, NLON, SAMPLE_SUBMISSION

N_YEARS = 86
N_ROWS = N_YEARS * NLAT * NLON


def to_frame(pred: np.ndarray) -> pd.DataFrame:
    """pred [86, 15, 192*288] -> submission frame (rows ordered by year, then latitude, then longitude)."""
    assert pred.shape == (N_YEARS, len(INDEX_NAMES), NLAT * NLON), pred.shape
    data = np.ascontiguousarray(pred.transpose(0, 2, 1)).reshape(N_ROWS, len(INDEX_NAMES))
    df = pd.DataFrame(data.astype(np.float32), columns=list(INDEX_NAMES))
    df.insert(0, "row_id", np.arange(N_ROWS, dtype=np.int64))
    return df


def check_frame(df: pd.DataFrame) -> None:
    """Raise if the frame would not be accepted: columns and row ids as in sample_submission.parquet,
    all predictions numeric and finite."""
    sample_cols = pq.ParquetFile(SAMPLE_SUBMISSION).schema_arrow.names
    sample_rows = pq.ParquetFile(SAMPLE_SUBMISSION).metadata.num_rows
    if list(df.columns) != list(sample_cols):
        raise ValueError(f"columns differ from the sample: {list(df.columns)}")
    if len(df) != sample_rows or len(df) != N_ROWS:
        raise ValueError(f"{len(df)} rows, expected {sample_rows}")
    if not np.array_equal(df["row_id"].to_numpy(), np.arange(N_ROWS)):
        raise ValueError("row_id is not 0..N-1 in order")
    vals = df[list(INDEX_NAMES)].to_numpy(np.float64)
    if not np.isfinite(vals).all():
        raise ValueError("non-finite predictions")


def write(pred: np.ndarray, path: Path) -> Path:
    df = to_frame(pred)
    check_frame(df)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False, compression="zstd", engine="pyarrow")
    back = pq.ParquetFile(tmp)
    assert back.metadata.num_rows == N_ROWS
    os.replace(tmp, path)
    return path
