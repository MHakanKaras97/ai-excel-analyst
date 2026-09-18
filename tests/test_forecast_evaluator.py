import pandas as pd

from src.forecasting.evaluator import evaluate_forecast_method


def _series(values, start="2024-01"):
    periods = pd.period_range(start=start, periods=len(values), freq="M")
    return pd.Series(values, index=[str(p) for p in periods])


def test_normal_history_naive_evaluation():
    series = _series([100.0, 100.0, 100.0, 100.0, 100.0, 90.0, 110.0])

    result = evaluate_forecast_method(series, method="naive", holdout=2)

    assert result["insufficient_data"] is False
    assert result["mae"] is not None
    assert result["rmse"] is not None
    assert result["mape"] is not None


def test_short_history_is_insufficient():
    series = _series([100.0, 110.0])

    result = evaluate_forecast_method(series, method="naive", holdout=3)

    assert result["insufficient_data"] is True
    assert result["reason"] == "insufficient_history"
    assert result["mae"] is None


def test_constant_series_gives_zero_error():
    series = _series([100.0] * 8)

    result = evaluate_forecast_method(series, method="naive", holdout=2)

    assert result["insufficient_data"] is False
    assert result["mae"] == 0.0
    assert result["rmse"] == 0.0
    assert result["mape"] == 0.0


def test_missing_values_in_holdout_is_rejected():
    series = _series([100.0, 110.0, 120.0, None, 140.0])

    result = evaluate_forecast_method(series, method="naive", holdout=2)

    assert result["insufficient_data"] is True
    assert result["reason"] == "missing_values_in_holdout"


def test_missing_values_in_training_is_rejected_not_silently_nan():
    # The NaN sits in the training portion (not the holdout) — must be
    # caught explicitly rather than flowing through into a silent `nan`
    # MAE/RMSE (float(nan) never raises on its own).
    series = _series([100.0, None, 120.0, 130.0, 140.0])

    result = evaluate_forecast_method(series, method="naive", holdout=2)

    assert result["insufficient_data"] is True
    assert result["reason"] == "missing_values_in_training"
    assert result["mae"] is None
    assert result["rmse"] is None
    assert result["mape"] is None


def test_invalid_holdout_is_rejected():
    series = _series([100.0, 110.0, 120.0])

    result = evaluate_forecast_method(series, method="naive", holdout=0)

    assert result["insufficient_data"] is True
    assert result["reason"] == "invalid_holdout"


def test_insufficient_history_for_seasonal_naive():
    series = _series([100.0] * 8)

    result = evaluate_forecast_method(series, method="seasonal_naive", holdout=2, season_length=12)

    assert result["insufficient_data"] is True
    assert result["reason"] == "insufficient_history"


def test_seasonal_naive_normal_history():
    series = _series([100.0, 90.0, 80.0] * 6)  # 18 months

    result = evaluate_forecast_method(series, method="seasonal_naive", holdout=3, season_length=12)

    assert result["insufficient_data"] is False
    assert result["mae"] == 0.0


def test_unsupported_method_is_rejected():
    series = _series([100.0, 110.0, 120.0])

    result = evaluate_forecast_method(series, method="linear_regression", holdout=1)

    assert result["insufficient_data"] is True
    assert result["reason"] == "unsupported_method"


def test_mape_is_none_when_all_actuals_are_zero():
    series = _series([0.0, 0.0, 0.0, 0.0, 0.0])

    result = evaluate_forecast_method(series, method="naive", holdout=2)

    assert result["insufficient_data"] is False
    assert result["mape"] is None
    assert result["mae"] == 0.0


def test_none_series_is_insufficient():
    result = evaluate_forecast_method(None, method="naive", holdout=1)

    assert result["insufficient_data"] is True
