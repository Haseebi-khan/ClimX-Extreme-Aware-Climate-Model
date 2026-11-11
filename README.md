# ClimX: extreme-aware climate emulation

Work for the Kaggle competition [ClimX main track](https://www.kaggle.com/competitions/climx-main-track):
emulate daily NorESM2-MM fields for SSP2-4.5 (2015-2100) from forcings; the score is the region-wise nNSE of
15 annual extreme indices computed from those fields.

Status and reasoning are in [DECISIONS.md](DECISIONS.md).

## Method in one paragraph

Each realisation of a target year is a whole simulated year from the training runs (historical, SSP1-2.6,
SSP3-7.0, SSP5-8.5), drawn with a Gaussian kernel on the forced global warming level and shifted, cell by cell
and day by day, to the warming level of the target year. The warming level of the target scenario comes from
its forcings through a small response model fitted on the training runs. The 15 indices are computed on every
realisation and averaged without weights. The last year is 364 days long, as in the data.

## Results (leave one scenario out, competition metric)

| emulator | SSP1-2.6 | SSP3-7.0 | SSP5-8.5 |
|---|---|---|---|
| v0: simulated years drawn by warming level | 0.036 | 0.167 | 0.181 |
| v1: v0 + daily temperatures shifted to the target warming level | 0.041 | 0.174 | 0.215 |
| oracle fitted to the held-out run itself (upper bound) | 0.060 | 0.197 | 0.244 |

## Reproduce

Python environment: `ml` (PyTorch for ROCm) plus `xarray netCDF4 h5netcdf dask cftime bottleneck numba zarr
xclim==0.59.1`. One GPU with at least 48 GB for the full-grid runs.

```bash
hf download isp-uv-es/ClimX --repo-type dataset --include "full/*" --local-dir data/hf   # 171 GB
kaggle competitions download -c climx-main-track -p data/kaggle                            # sample submission

python scripts/build_store.py            # NetCDF -> flat float32 arrays (work/store, 137 GB)
python scripts/build_thresholds.py       # day-of-year percentile thresholds from historical 1961-1990
python scripts/verify_indices.py         # GPU indices vs the xclim reference
python tests/test_metric.py              # array scorer vs the organisers' rnNSE
python scripts/build_truth_indices.py    # indices of every training run
python scripts/build_annual_means.py     # annual means and global-mean temperature
python scripts/build_index_tables.py     # per-year index tables (for v0)

python scripts/ceiling.py                # what is achievable on this metric
python scripts/validate_v0.py            # leave-one-scenario-out, v0
python scripts/validate_v1.py --stride 4 --configs sweep2
python scripts/make_submission_v1.py --h 0.5 --draws 200 --holdout ssp370   # end-to-end check
python scripts/make_submission_v1.py --h 0.5 --draws 200                    # SSP2-4.5 submission file
```

## Layout

```
src/climx/     indices.py (GPU indices), reference_xclim.py, metric.py, official_rnNSE.py (organisers' scorer,
               unchanged), forcing.py (warming-level emulator), analog.py, resample.py (v0), engine.py (v1),
               data.py, submission.py, paths.py
scripts/       data preparation, verification, validation and submission scripts
tests/         test_metric.py
data/, work/, external/, submissions/*.parquet    not in git
```

`src/climx/official_rnNSE.py` and the index definitions transcribed in `src/climx/reference_xclim.py` come from
the organisers' repository (https://github.com/IPL-UV/ClimX, MIT licence, see LICENSES/ClimX-MIT.txt).
