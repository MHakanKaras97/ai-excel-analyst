import pandas as pd

from src.advanced_analytics.statistics import growth_rate, rolling_mean, rolling_std, seasonality_signal, volatility


def _series(values, start="2024-01"):
    periods = pd.period_range(start=start, periods=len(values), freq="M")
    return pd.Series(values, index=[str(p) for p in periods])


# ==================================================
# rolling_mean / rolling_std
# ==================================================


def test_rolling_mean_normal_history():
    series = _series([10.0, 20.0, 30.0, 40.0])

    result = rolling_mean(series, window=2)

    assert result["insufficient_data"] is False
    assert result["values"] == [15.0, 25.0, 35.0]
    assert result["periods"] == ["2024-02", "2024-03", "2024-04"]


def test_rolling_mean_insufficient_history():
    series = _series([10.0, 20.0])

    result = rolling_mean(series, window=3)

    assert result["insufficient_data"] is True
    assert result["values"] == []


def test_rolling_mean_none_series_is_insufficient():
    assert rolling_mean(None)["insufficient_data"] is True


def test_rolling_std_constant_series_is_zero_not_none():
    series = _series([10.0, 10.0, 10.0, 10.0])

    result = rolling_std(series, window=2)

    assert result["insufficient_data"] is False
    assert all(v == 0.0 for v in result["values"])


# ==================================================
# growth_rate
# ==================================================


def test_growth_rate_normal_history():
    series = _series([100.0, 110.0, 121.0])

    result = growth_rate(series)

    assert result["insufficient_data"] is False
    assert result["rates"] == [10.0, 10.0]


def test_growth_rate_insufficient_history():
    series = _series([100.0])

    result = growth_rate(series)

    assert result["insufficient_data"] is True
    assert result["rates"] == []


def test_growth_rate_handles_zero_previous_value():
    series = _series([0.0, 50.0])

    result = growth_rate(series)

    assert result["insufficient_data"] is False
    assert result["rates"] == [None]


# ==================================================
# volatility
# ==================================================


def test_volatility_normal_history():
    series = _series([100.0, 110.0, 90.0, 120.0])

    result = volatility(series)

    assert result["insufficient_data"] is False
    assert result["value"] > 0


def test_volatility_constant_series_is_zero():
    series = _series([100.0, 100.0, 100.0])

    result = volatility(series)

    assert result["insufficient_data"] is False
    assert result["value"] == 0.0


def test_volatility_short_history_is_insufficient():
    series = _series([100.0, 110.0])

    result = volatility(series)

    assert result["insufficient_data"] is True
    assert result["value"] is None


def test_volatility_missing_values_insufficient():
    series = _series([0.0, 50.0])  # only produces one valid rate (None from div-by-zero excluded)

    result = volatility(series)

    assert result["insufficient_data"] is True


# ==================================================
# seasonality_signal
# ==================================================


def test_seasonality_signal_insufficient_with_short_history():
    series = _series([100.0] * 6)

    result = seasonality_signal(series)

    assert result["insufficient_data"] is True
    assert result["seasonal_strength"] is None


def test_seasonality_signal_with_two_years_of_data():
    values = [100.0, 90.0, 80.0, 150.0, 100.0, 90.0, 80.0, 150.0, 100.0, 90.0, 80.0, 150.0,
              100.0, 90.0, 80.0, 150.0, 100.0, 90.0, 80.0, 150.0, 100.0, 90.0, 80.0, 150.0]
    series = _series(values)

    result = seasonality_signal(series)

    assert result["insufficient_data"] is False
    assert result["peak_month"] == 4
    assert result["trough_month"] == 3
    assert result["seasonal_strength"] > 0


def test_seasonality_signal_constant_series_reports_insufficient_due_to_zero_variation():
    series = _series([100.0] * 24)

    result = seasonality_signal(series)

    assert result["insufficient_data"] is False
    assert result["seasonal_strength"] == 0.0


def test_seasonality_signal_none_series_is_insufficient():
    assert seasonality_signal(None)["insufficient_data"] is True


def test_seasonality_signal_invalid_period_labels_are_insufficient():
    series = pd.Series([1.0] * 24, index=[f"row{i}" for i in range(24)])

    result = seasonality_signal(series)

    assert result["insufficient_data"] is True


def test_seasonality_signal_malformed_yyyy_m_label_is_rejected_not_misparsed():
    # Regression test: "2024-1" (single-digit month, not the project's
    # zero-padded "YYYY-MM" contract) must be rejected outright rather than
    # silently misparsed via a bare string slice.
    labels = [f"2024-{m:02d}" for m in range(1, 13)] * 2
    labels[0] = "2024-1"
    series = pd.Series([1.0] * 24, index=labels)

    result = seasonality_signal(series)

    assert result["insufficient_data"] is True
