"""Deterministic forecast backtesting/evaluation (V0.8.11).

Holds out the last `holdout` periods, forecasts them from the remaining
history using the requested baseline, and compares against the actual
values. MAE/RMSE are always computable once a full holdout forecast
exists; MAPE is only computed over holdout points with a non-zero actual
value (its own assumption), and is `None` — never fabricated — when no
such point exists.
"""
import numpy as np

from src.forecasting.baseline import naive_forecast, seasonal_naive_forecast

_METHODS = {"naive": naive_forecast, "seasonal_naive": seasonal_naive_forecast}


def _failure(reason: str, method: str) -> dict:
    return {"insufficient_data": True, "reason": reason, "method": method, "mae": None, "rmse": None, "mape": None}


def evaluate_forecast_method(series, method: str = "naive", holdout: int = 3, season_length: int = 12) -> dict:
    if method not in _METHODS:
        return _failure("unsupported_method", method)
    if holdout < 1:
        return _failure("invalid_holdout", method)
    if series is None or len(series) <= holdout:
        return _failure("insufficient_history", method)

    train = series.iloc[:-holdout]
    actual = series.iloc[-holdout:]

    if train.isna().any():
        # A NaN in the training history could otherwise flow through into a
        # silent `nan` MAE/RMSE (float(nan) never raises) instead of an
        # explicit refusal — reject before any baseline call is made.
        return _failure("missing_values_in_training", method)

    kwargs = {"periods_ahead": holdout}
    if method == "seasonal_naive":
        kwargs["season_length"] = season_length

    forecast_result = _METHODS[method](train, **kwargs)
    if forecast_result["insufficient_data"] or len(forecast_result["forecasts"]) != holdout:
        return _failure("insufficient_history", method)

    predicted = np.array([f["forecast"] for f in forecast_result["forecasts"]], dtype=float)
    actual_values = actual.to_numpy(dtype=float)
    if np.isnan(actual_values).any():
        return _failure("missing_values_in_holdout", method)

    errors = actual_values - predicted
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    nonzero_mask = actual_values != 0
    mape = float(np.mean(np.abs(errors[nonzero_mask] / actual_values[nonzero_mask])) * 100) if nonzero_mask.any() else None

    return {
        "insufficient_data": False, "reason": None, "method": method,
        "holdout": holdout, "mae": mae, "rmse": rmse, "mape": mape,
    }
