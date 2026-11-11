# Decisions

Dated log of decisions, the reasoning and the evidence. Scores are the competition metric (region-wise nNSE).
"Held out" means the scenario's outputs were not used in any way; its forcings were.

## 2026-11-10 — What the competition asks and allows

- Task: emulate daily NorESM2-MM fields (7 variables, 192 x 288) for SSP2-4.5, 2015-2100, from forcings.
  Scored on 15 annual extreme indices computed from the daily fields, per AR6 land cell.
- The indices must be computed from the emulator's daily fields. The only operation allowed on index maps is
  an unweighted mean over realisations of the same emulator, with the same draws for every index and cell
  (repeats allowed). No smoothing, bias correction or blending of index maps (organisers, topics 748292,
  750174).
- Allowed: training data at inference (752230); realisations built from simulated training years (748292);
  transforms of daily fields fitted on ClimX data; physical structure with parameters fitted on ClimX data
  only (751312). Not allowed: CMIP6 or other external climate data, pretrained weather/climate models,
  externally calibrated parameters.
- Final ranking = public leaderboard at the deadline (15 Dec 2026), no hidden split (756766). Two submissions
  per day. Top teams must release code and weights under MIT or Apache-2.0 and pass a reproducibility check
  in which the organisers recompute the indices from our daily fields with their code.

## 2026-11-10 — The 0.95 target is not on this metric's scale

Evidence, all from the training scenarios (scripts/ceiling.py):

| predictor (fitted to the held-out run's own indices) | SSP1-2.6 | SSP3-7.0 | SSP5-8.5 |
|---|---|---|---|
| per-cell cubic in time, noise-adjusted | 0.015 | 0.150 | 0.198 |
| the same plus a separate value for the 364-day year 2100 | 0.060 | 0.197 | 0.244 |

- The metric is the share of each cell's year-to-year index variance that is explained. Most of that variance
  is internal variability, which forcings do not determine. Daily weather is uncorrelated between scenarios
  from the first days of 2015 (correlation about 0.01).
- The oracle is optimistic: a cubic also fits slow internal variability.
- SSP2-4.5 warms 0.78 -> 2.24 K (emulated), between SSP1-2.6 (0.87 -> 1.34) and SSP3-7.0 (0.74 -> 3.35), so
  its ceiling lies between 0.06 and 0.20; about 0.10-0.14.
- Leaderboard on 2026-11-10: leader 0.1246, pack around 0.105, organisers' baselines -0.45 to -0.57.

Decision: aim for first place (above 0.125), not for 0.95.

## 2026-11-10 — Exact reproduction of indices and metric

- `src/climx/indices.py` computes the 15 indices on the GPU. Checked against an xclim reference
  (`src/climx/reference_xclim.py`, a call-for-call transcription of the organisers' `indices_xclim.py`) on
  3,000 cells x 4 runs: zero mismatches in all 60 index/run combinations (`scripts/verify_indices.py`).
  The check is against this transcription; running the organisers' file itself on the same cells is still
  to be done.
- Percentile thresholds (WSDI, CSDI, R95p) come from xclim `percentile_doy` on historical 1961-1990, kept in
  float64 as in the official comparison; rounding them to float32 changed 17 of 61,920 WSDI values.
- `src/climx/metric.py` equals the organisers' `rnNSE.score` to 1e-9 (`tests/test_metric.py`).

## 2026-11-10 — The year 2100 has 364 days

Every file ends one day early (31,389 days), including the SSP2-4.5 time axis; the sample submission shows
364 in the last year. Cells where an index equals the number of days in the year (frost days in Antarctica,
growing season in the tropics, ...) therefore vary only through that year and are perfectly predictable: 46 %
of the scored GSL cells, about 25 % for FD and ID, 11 % for TR. Worth about +0.045 in total. The emulator
generates the given time axis, so its last year is 364 days long.

## 2026-11-10 — Validation

Leave one scenario out: fit everything without one training SSP, predict it, score with the exact metric.
SSP1-2.6 (weak signal) and SSP3-7.0 (strong signal) bracket the test scenario, so their mean is the selection
criterion. SSP5-8.5 is reported but its last decades are an extrapolation in warming level that SSP2-4.5 does
not need. Sweeps use every 4th scored cell; final numbers use all cells.

## 2026-11-10/11 — Emulators

Forced warming level: annual global-mean temperature of each training run, low-pass filtered; for the target,
a one-box response to log CO2 fitted on the training runs (`src/climx/forcing.py`), error 0.07-0.10 K when a
scenario is held out. Using the held-out run's true warming level instead changes the score by +0.0006.

| emulator | SSP1-2.6 | SSP3-7.0 | SSP5-8.5 | mean 126/370 |
|---|---|---|---|---|
| v0: simulated years drawn by warming level (Gaussian kernel, h = 0.3 K), unchanged | 0.036 | 0.167 | 0.181 | 0.102 |
| v1: v0 + daily temperatures shifted to the target warming level, h = 0.5 K | 0.041 | 0.174 | 0.215 | 0.107 |

(v0 and v1 full-grid numbers for SSP3-7.0 were confirmed with the organisers' scorer: 0.1672 and 0.1742.)

Tried and not adopted (mean 126/370, every 4th cell; v1 = 0.1074):

- rescaling temperature anomalies by the ratio of standard deviations: 0.1074 (no change at h = 0.5)
- scaling precipitation by the ratio of seasonal means: no gain
- aerosol, CO2, CH4 or quadratic terms in the shift model: 0.101-0.107 (worse on SSP1-2.6)
- choosing analog years that also match aerosol or CO2 levels: at most +0.0015 for v0
- wider kernels (h = 0.8, 1.5): worse for frost days, better for precipitation, worse overall

What is left (`scripts/diagnose_error.py`): the largest systematic error is a constant per-cell offset between
scenarios at equal global warming (2-6 % of the index variance), larger than the trend error. SSP3-7.0 warms
northern land about 0.1 K less than the others at equal global warming (aerosols).

## 2026-11-11 — v1 is on a plateau; what the remaining error is

- Kernel width 0.4-0.7 K, 200 or 800 realisations, with or without the early historical years: all within
  0.1068-0.1078 (mean 126/370, every 4th cell). Chosen: h = 0.5 K, 200 realisations.
- The offset is not noise of slow internal variability: index anomalies in the 165-year historical run are
  close to white (lag-1 autocorrelation about 0.1 for temperature indices, 0 for precipitation).
- Part of it is the emulated warming level of the target. With a scenario held out the emulator is off by
  +0.07, +0.10 and -0.11 K (SSP1-2.6, 3-7.0, 5-8.5), which moves the land-mean temperature by the same sign
  (+0.05, +0.10, -0.11 K). With the run's own smoothed warming level the land-mean error drops to 0.03-0.04 K,
  but the cell-level offset stays at about 9 % of the interannual variance of annual-mean temperature, and
  the score changes by only +0.0006. For SSP2-4.5 all three SSPs are used for the fit and its forcing lies
  inside their range, so the emulator error should be smaller than these held-out values.
- Regional aerosol terms (SO2 or BC emission maps smoothed over 700-3000 km, linear or log) lower the offset
  for SSP3-7.0 by about 15 % and raise it for SSP1-2.6 (`scripts/offset_study.py`). Not adopted.

## 2026-11-11 — First leaderboard scores

| file | held out SSP1-2.6 | held out SSP3-7.0 | mean | leaderboard (SSP2-4.5) |
|---|---|---|---|---|
| `v0_h0.3_n200.parquet` | 0.036 | 0.167 | 0.102 | 0.0975 |
| `v1_shift_h0.5_n200.parquet` | 0.041 | 0.174 | 0.107 | 0.1012 |

- The leaderboard score is about 0.005 below the mean of the two bracketing held-out scores, for both files.
- Validation orders the two emulators correctly: v1 - v0 is +0.006 in validation and +0.004 on the leaderboard.
- 0.1012 is rank 33 of 188. Leader 0.1246, second 0.1098, 14 teams between 0.105 and 0.110.

## Open

- Closing the gap to the 0.105-0.110 group and to the leader (0.1246).
- Licence of this repository (MIT or Apache-2.0 is required for prize eligibility): to be chosen by the team.
- A writer for the daily fields of the realisations, for the reproducibility check.
