import pandas as pd

from src.forecasting.baseline import naive_forecast, seasonal_naive_forecast


def _series(values, start="2024-01"):
    periods = pd.period_range(start=start, periods=len(values), freq="M")
    return pd.Series(values, index=[str(p) for p in periods])


# ==================================================
# naive_forecast
# ==================================================


def test_naive_forecast_normal_history():
    series = _series([100.0, 110.0, 105.0, 120.0])

    result = naive_forecast(series, periods_ahead=2)

    assert result["insufficient_data"] is False
    forecasts = result["forecasts"]
    assert len(forecasts) == 2
    assert forecasts[0]["period"] == "2024-05"
    assert forecasts[1]["period"] == "2024-06"
    assert all(f["forecast"] == 120.0 for f in forecasts)
    assert all(f["is_forecast"] is True for f in forecasts)
    assert all(f["bounds_available"] is True for f in forecasts)
    assert forecasts[0]["lower_bound"] < 120.0 < forecasts[0]["upper_bound"]


def test_naive_forecast_short_history_has_no_bounds():
    series = _series([100.0])

    result = naive_forecast(series)

    assert result["insufficient_data"] is False
    assert result["forecasts"][0]["bounds_available"] is False
    assert result["forecasts"][0]["lower_bound"] is None
    assert result["forecasts"][0]["upper_bound"] is None


def test_naive_forecast_constant_series_has_zero_width_bounds():
    series = _series([100.0, 100.0, 100.0])

    result = naive_forecast(series)

    forecast = result["forecasts"][0]
    assert forecast["bounds_available"] is True
    assert forecast["lower_bound"] == forecast["upper_bound"] == 100.0


def test_naive_forecast_empty_series_is_insufficient():
    series = _series([])

    result = naive_forecast(series)

    assert result["insufficient_data"] is True
    assert result["forecasts"] == []


def test_naive_forecast_none_series_is_insufficient():
    result = naive_forecast(None)

    assert result["insufficient_data"] is True


def test_naive_forecast_distinguishes_prediction_from_fact():
    series = _series([100.0, 110.0])

    result = naive_forecast(series)

    assert result["forecasts"][0]["is_forecast"] is True


def test_naive_forecast_nan_last_value_is_insufficient_not_silently_nan():
    series = _series([100.0, 110.0, float("nan")])

    result = naive_forecast(series)

    assert result["insufficient_data"] is True
    assert result["forecasts"] == []


def test_naive_forecast_nan_earlier_in_history_does_not_block_a_valid_tail():
    # A NaN not at the tail must not prevent forecasting from the last
    # valid observation — existing valid-data behavior must be unaffected.
    series = _series([100.0, float("nan"), 110.0, 120.0])

    result = naive_forecast(series)

    assert result["insufficient_data"] is False
    assert result["forecasts"][0]["forecast"] == 120.0


# ==================================================
# seasonal_naive_forecast
# ==================================================


def test_seasonal_naive_forecast_normal_history():
    values = [10.0, 20.0, 30.0] * 4 + [15.0, 25.0, 35.0]  # 15 months
    series = _series(values)

    result = seasonal_naive_forecast(series, periods_ahead=1, season_length=12)

    assert result["insufficient_data"] is False
    # 12 months back from the next forecasted period (2025-04) is series[3] = "2024-04" = 10.0
    assert result["forecasts"][0]["forecast"] == 10.0


def test_seasonal_naive_forecast_insufficient_history():
    series = _series([1.0] * 6)

    result = seasonal_naive_forecast(series, season_length=12)

    assert result["insufficient_data"] is True
    assert result["forecasts"] == []


def test_seasonal_naive_forecast_stops_after_one_season_of_anchors():
    series = _series(list(range(12)), start="2024-01")

    result = seasonal_naive_forecast(series, periods_ahead=20, season_length=12)

    assert result["insufficient_data"] is False
    assert len(result["forecasts"]) == 12


def test_seasonal_naive_forecast_none_series_is_insufficient():
    assert seasonal_naive_forecast(None)["insufficient_data"] is True


def test_seasonal_naive_forecast_constant_series_zero_width_bounds():
    series = _series([50.0] * 26)

    result = seasonal_naive_forecast(series, season_length=12)

    forecast = result["forecasts"][0]
    assert forecast["bounds_available"] is True
    assert forecast["lower_bound"] == forecast["upper_bound"] == 50.0


def test_seasonal_naive_forecast_skips_nan_anchor_but_keeps_later_valid_steps():
    # With 13 months of history and season_length=12, step 1's anchor is
    # series[1] and step 2's anchor is series[2] — make the first NaN and
    # the second valid, so step 1 must be skipped without blocking step 2.
    values = [0.0, float("nan"), 99.0] + [0.0] * 10
    series = _series(values)

    result = seasonal_naive_forecast(series, periods_ahead=2, season_length=12)

    assert result["insufficient_data"] is False
    assert len(result["forecasts"]) == 1
    assert result["forecasts"][0]["forecast"] == 99.0


def test_seasonal_naive_forecast_all_anchors_nan_is_insufficient():
    values = [float("nan")] * 12
    series = _series(values)

    result = seasonal_naive_forecast(series, periods_ahead=1, season_length=12)

    assert result["insufficient_data"] is True
    assert result["forecasts"] == []
