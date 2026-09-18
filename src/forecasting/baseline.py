"""Naive and seasonal-naive forecast baselines (V0.8.10).

Deliberately the only two forecasting methods implemented — both are
simple, explainable, and don't require fitting any model. Confidence
bounds are derived from the actual historical residual spread (consecutive
differences for `naive`, same-season differences for `seasonal_naive`) and
are only reported when there is enough history to compute a defensible
standard deviation; otherwise `bounds_available` is False and no interval
is fabricated.
"""
import numpy as np
import pandas as pd

from src.forecasting.models import CONFIDENCE_Z, MIN_RESIDUALS_FOR_BOUNDS, make_forecast_entry


def _next_period_label(label: str, steps: int) -> str:
    return str(pd.Period(label, freq="M") + steps)


def _bounds(value: float, residual_std, bounds_available: bool):
    if not bounds_available:
        return None, None
    margin = CONFIDENCE_Z * residual_std
    return value - margin, value + margin


def naive_forecast(series: pd.Series, periods_ahead: int = 1) -> dict:
    """Flat forecast: every future period repeats the last observed value."""
    if series is None or len(series) < 1 or periods_ahead < 1:
        return {"insufficient_data": True, "method": "naive", "forecasts": []}

    if pd.isna(series.iloc[-1]):
        # The one value this method would carry forward is unknown — every
        # future step would rely on it, so there is no partial result to
        # salvage here (unlike seasonal_naive_forecast, where only some
        # steps might be affected).
        return {"insufficient_data": True, "method": "naive", "forecasts": []}

    last_value = float(series.iloc[-1])
    diffs = series.diff().dropna().to_numpy()
    bounds_available = len(diffs) >= MIN_RESIDUALS_FOR_BOUNDS
    residual_std = float(np.std(diffs, ddof=1)) if bounds_available else None

    last_label = series.index[-1]
    forecasts = []
    for step in range(1, periods_ahead + 1):
        lower, upper = _bounds(last_value, residual_std, bounds_available)
        forecasts.append(make_forecast_entry(
            _next_period_label(last_label, step), last_value, "naive",
            bounds_available, lower, upper,
        ))

    return {"insufficient_data": False, "method": "naive", "forecasts": forecasts}


def seasonal_naive_forecast(series: pd.Series, periods_ahead: int = 1, season_length: int = 12) -> dict:
    """Each future period repeats the value from exactly one season ago.
    Requires at least one full season of history; a request for more
    periods ahead than one season can anchor to (i.e. beyond the first
    unseen season) stops rather than guessing further.
    """
    if series is None or len(series) < season_length or periods_ahead < 1:
        return {"insufficient_data": True, "method": "seasonal_naive", "forecasts": []}

    seasonal_diffs = (series - series.shift(season_length)).dropna().to_numpy()
    bounds_available = len(seasonal_diffs) >= MIN_RESIDUALS_FOR_BOUNDS
    residual_std = float(np.std(seasonal_diffs, ddof=1)) if bounds_available else None

    last_label = series.index[-1]
    forecasts = []
    for step in range(1, periods_ahead + 1):
        source_position = len(series) - season_length + (step - 1)
        if not (0 <= source_position < len(series)):
            break
        anchor_value = series.iloc[source_position]
        if pd.isna(anchor_value):
            # This step's anchor is unknown; later steps may still have a
            # valid anchor of their own, so skip rather than stop entirely.
            continue
        value = float(anchor_value)
        lower, upper = _bounds(value, residual_std, bounds_available)
        forecasts.append(make_forecast_entry(
            _next_period_label(last_label, step), value, "seasonal_naive",
            bounds_available, lower, upper,
        ))

    if not forecasts:
        return {"insufficient_data": True, "method": "seasonal_naive", "forecasts": []}

    return {"insufficient_data": False, "method": "seasonal_naive", "forecasts": forecasts}
