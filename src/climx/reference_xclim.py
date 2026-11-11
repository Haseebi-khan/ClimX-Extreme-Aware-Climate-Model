"""Reference implementation of the 15 ClimX indices with xclim.

This file transcribes, call for call, what the organisers' ``src/indices_xclim.py`` (MIT licence,
https://github.com/IPL-UV/ClimX, commit 142de09) does for the 15 leaderboard indices, followed by the
annual max/min that ``format_indices`` applies to the monthly TXx, TNn and Rx5day. It is slow and is used
only to compute the percentile thresholds and to verify the GPU implementation in ``indices.py``.

Inputs are xarray DataArrays with a daily noleap ``time`` axis: temperatures in K, precipitation in
kg m-2 s-1 (as in the dataset).
"""
from __future__ import annotations

import numpy as np
import xarray as xr
import xclim.indices as xci
from xclim.core.calendar import percentile_doy
from xclim.indices import run_length as rl

KG_M2_S_TO_MM_DAY = 86400
BASE_PERIOD = ("1961-01-01", "1990-12-31")
INDEX_NAMES = ("TXx", "TNn", "SU", "TR", "FD", "ID", "WSDI", "CSDI", "GSL",
               "Rx5day", "CDD", "CWD", "R95p", "R10mm", "SDII")


def percentile_threshold(da_hist: xr.DataArray, quantile: float, base_period=BASE_PERIOD,
                         window: int = 5) -> xr.DataArray:
    """Day-of-year percentile over the base period (official ``_percentile_threshold``)."""
    da = da_hist.sel(time=slice(base_period[0], base_period[1]))
    return percentile_doy(da, window=window, per=quantile).squeeze("percentiles", drop=True)


def precip_mm_day(da: xr.DataArray) -> xr.DataArray:
    converted = da * KG_M2_S_TO_MM_DAY
    converted.attrs.update(da.attrs)
    converted.attrs["units"] = "mm/day"
    return converted


def _align_dayofyear(da: xr.DataArray, threshold: xr.DataArray) -> xr.DataArray:
    return threshold.sel(dayofyear=da["time"].dt.dayofyear)


def _spell_days(mask: xr.DataArray, min_length: int) -> xr.DataArray:
    """Per calendar year, the number of days in runs of at least ``min_length`` days."""
    years = np.unique(mask["time"].dt.year.values)
    outputs = []
    for year in years:
        subset = mask.sel(time=mask["time"].dt.year == year)
        stat = rl.windowed_run_count(subset, window=min_length, dim="time")
        outputs.append(stat.expand_dims(year=[int(year)]).rename(None))
    return xr.concat(outputs, dim="year")


def _annual(da: xr.DataArray, how: str = "first") -> xr.DataArray:
    g = da.groupby("time.year")
    return {"first": g.first, "max": lambda: g.max("time"), "min": lambda: g.min("time")}[how]()


def compute_reference(tasmax: xr.DataArray, tasmin: xr.DataArray, tas: xr.DataArray, pr: xr.DataArray,
                      thr_tx90: xr.DataArray, thr_tn10: xr.DataArray, thr_pr95_raw: xr.DataArray
                      ) -> dict[str, xr.DataArray]:
    """The 15 annual indices, each with a ``year`` dimension. Thresholds come from
    ``percentile_threshold`` on the historical run (``thr_pr95_raw`` in kg m-2 s-1)."""
    pr_mm = precip_mm_day(pr)
    out = {}
    out["FD"] = _annual(xci.generic.threshold_count(tasmin, threshold=273.15, op="<", freq="YS"))
    out["SU"] = _annual(xci.generic.threshold_count(tasmax, threshold=298.15, op=">", freq="YS"))
    out["ID"] = _annual(xci.generic.threshold_count(tasmax, threshold=273.15, op="<", freq="YS"))
    out["TR"] = _annual(xci.generic.threshold_count(tasmin, threshold=293.15, op=">", freq="YS"))
    out["GSL"] = _annual(xci.growing_season_length(tas=tas, thresh="5 degC", freq="YS"))
    out["TXx"] = _annual(xci.tx_max(tasmax=tasmax, freq="MS"), "max")
    out["TNn"] = _annual(xci.tn_min(tasmin=tasmin, freq="MS"), "min")
    out["WSDI"] = _spell_days(tasmax > _align_dayofyear(tasmax, thr_tx90), 6)
    out["CSDI"] = _spell_days(tasmin < _align_dayofyear(tasmin, thr_tn10), 6)
    out["Rx5day"] = _annual(xci.max_n_day_precipitation_amount(pr=pr_mm, window=5, freq="MS"), "max")
    out["CDD"] = _annual(xci.maximum_consecutive_dry_days(pr=pr_mm, thresh="1 mm/day", freq="YS"))
    out["CWD"] = _annual(xci.maximum_consecutive_wet_days(pr=pr_mm, thresh="1 mm/day", freq="YS"))
    mask = pr_mm.groupby("time.dayofyear") > precip_mm_day(thr_pr95_raw)
    out["R95p"] = _annual(pr_mm.where(mask).resample(time="YS").sum(dim="time"))
    wet = pr_mm.where(pr_mm >= 1)
    wet_sum = wet.resample(time="YS").sum(dim="time")
    wet_count = wet.resample(time="YS").count(dim="time")
    out["SDII"] = _annual(xr.where(wet_count == 0, 0, wet_sum / wet_count))
    out["R10mm"] = _annual(xci.wetdays(pr=pr_mm, thresh="10 mm/day", freq="YS"))
    return out
