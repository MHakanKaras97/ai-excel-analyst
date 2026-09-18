"""Deterministic advanced-analytics metrics (V0.8.9).

Each function takes an already-aggregated period Series (as produced by
file_analysis.build_monthly_series — index = "YYYY-MM" labels, values =
numeric). No LLM involvement, no new period/column resolution — this
module is purely arithmetic over an already-resolved series, reusing
analytics_engine.compare_periods() for the pairwise change math rather
than recomputing percentage change a second time.
"""
import numpy as np
import pandas as pd

from src.advanced_analytics.models import (
    DEFAULT_ROLLING_WINDOW,
    MIN_SEASONALITY_OBSERVATIONS,
    MIN_VOLATILITY_OBSERVATIONS,
    insufficient_result,
)
from src.analytics_engine import compare_periods
from src.qa_engine import PERIOD_LABEL_PATTERN


def rolling_mean(series: pd.Series, window: int = DEFAULT_ROLLING_WINDOW) -> dict:
    if series is None or len(series) < window:
        return insufficient_result(window=window, periods=[], values=[])

    rolled = series.rolling(window=window).mean().dropna()
    return {
        "insufficient_data": False, "window": window,
        "periods": [str(p) for p in rolled.index], "values": [float(v) for v in rolled.to_numpy()],
    }


def rolling_std(series: pd.Series, window: int = DEFAULT_ROLLING_WINDOW) -> dict:
    if series is None or len(series) < window:
        return insufficient_result(window=window, periods=[], values=[])

    rolled = series.rolling(window=window).std().dropna()
    return {
        "insufficient_data": False, "window": window,
        "periods": [str(p) for p in rolled.index], "values": [float(v) for v in rolled.to_numpy()],
    }


def growth_rate(series: pd.Series) -> dict:
    """Period-over-period percentage change, delegating the actual
    arithmetic entirely to analytics_engine.compare_periods()."""
    if series is None:
        return insufficient_result(periods=[], rates=[])

    comparison = compare_periods(series)
    if comparison["insufficient_data"]:
        return insufficient_result(periods=comparison["periods"], rates=[])

    return {
        "insufficient_data": False,
        "periods": [c["to_period"] for c in comparison["comparisons"]],
        "rates": [c["percentage_change"] for c in comparison["comparisons"]],
    }


def volatility(series: pd.Series) -> dict:
    """Sample standard deviation of period-over-period growth rates — a
    measure of how much the growth rate itself varies, not of the raw
    values. Requires at least MIN_VOLATILITY_OBSERVATIONS valid growth
    rates (a constant series or one with too few valid transitions
    reports insufficient_data rather than a fabricated 0.0/None value).
    """
    rates = growth_rate(series)
    if rates["insufficient_data"]:
        return insufficient_result(value=None)

    valid_rates = [r for r in rates["rates"] if r is not None]
    if len(valid_rates) < MIN_VOLATILITY_OBSERVATIONS:
        return insufficient_result(value=None)

    return {"insufficient_data": False, "value": float(np.std(valid_rates, ddof=1))}


def _month_of_label(label: str):
    """Extract the calendar month from a "YYYY-MM" period label, validated
    against the same PERIOD_LABEL_PATTERN qa_engine.py already uses for
    period resolution — rather than a bare slice, which would silently
    misparse a malformed label (e.g. "2024-1") instead of rejecting it."""
    label_str = str(label)
    if not PERIOD_LABEL_PATTERN.match(label_str):
        return None
    return int(label_str[5:7])


def seasonality_signal(series: pd.Series) -> dict:
    """A conservative signal of monthly seasonality: the coefficient of
    variation of per-calendar-month averages, plus which calendar month
    tends to be highest/lowest. This is NOT a seasonal decomposition (no
    STL/Fourier fitting) — it is a simple, defensible summary that requires
    at least MIN_SEASONALITY_OBSERVATIONS (two full years) of monthly data,
    and refuses to report a signal from anything less.
    """
    if series is None or len(series) < MIN_SEASONALITY_OBSERVATIONS:
        return insufficient_result(seasonal_strength=None, peak_month=None, trough_month=None)

    months = [_month_of_label(label) for label in series.index]
    if any(month is None for month in months):
        return insufficient_result(seasonal_strength=None, peak_month=None, trough_month=None)

    by_month = pd.Series(series.to_numpy(), index=months).groupby(level=0).mean()
    if len(by_month) < 2:
        return insufficient_result(seasonal_strength=None, peak_month=None, trough_month=None)

    overall_mean = float(pd.Series(series.to_numpy()).mean())
    if overall_mean == 0:
        return insufficient_result(seasonal_strength=None, peak_month=None, trough_month=None)

    seasonal_strength = float(by_month.std(ddof=0) / abs(overall_mean))
    return {
        "insufficient_data": False,
        "seasonal_strength": seasonal_strength,
        "peak_month": int(by_month.idxmax()),
        "trough_month": int(by_month.idxmin()),
    }
