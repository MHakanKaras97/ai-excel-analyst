import pandas as pd
import pytest

from app import (
    ai_error_message,
    build_analytics_payload,
    build_monthly_series,
    build_period_series,
    date_like_column_names,
    format_column_option,
    numeric_column_names,
    save_uploaded_file,
)
from src.analytics_engine import compare_periods, detect_trend
from src.anomaly_detector import detect_iqr_anomalies


class FakeUploadedFile:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def getbuffer(self):
        return self._content


def test_save_uploaded_file_writes_bytes_and_preserves_extension(tmp_path):
    uploaded = FakeUploadedFile("report.xlsx", b"fake-excel-bytes")

    dest = save_uploaded_file(uploaded, tmp_path)

    assert dest.suffix == ".xlsx"
    assert dest.read_bytes() == b"fake-excel-bytes"
    assert dest.parent == tmp_path


def test_numeric_column_names_selects_only_numeric_dtypes():
    df = pd.DataFrame({
        "revenue": [1.0, 2.0, 3.0],
        "label": ["a", "b", "c"],
        "count": [1, 2, 3],
    })

    assert numeric_column_names(df) == [(0, "revenue"), (2, "count")]


def test_numeric_column_names_empty_when_no_numeric_columns():
    df = pd.DataFrame({"label": ["a", "b"]})

    assert numeric_column_names(df) == []


def test_numeric_column_names_handles_duplicate_column_names_positionally():
    # Two columns both named "revenue": position 0 is numeric, position 1 is not.
    df = pd.DataFrame([[10, "x"], [20, "y"]], columns=["revenue", "revenue"])

    result = numeric_column_names(df)

    assert result == [(0, "revenue")]


def test_date_like_column_names_filters_by_inferred_type():
    profile = {
        "columns": [
            {"name": "period", "inferred_type": "date-like"},
            {"name": "revenue", "inferred_type": "numeric"},
            {"name": "notes", "inferred_type": "text"},
        ]
    }

    assert date_like_column_names(profile) == [(0, "period")]


def test_date_like_column_names_empty_when_none_found():
    profile = {"columns": [{"name": "revenue", "inferred_type": "numeric"}]}

    assert date_like_column_names(profile) == []


def test_date_like_column_names_handles_duplicate_names_positionally():
    profile = {
        "columns": [
            {"name": "period", "inferred_type": "date-like"},
            {"name": "period", "inferred_type": "date-like"},
            {"name": "revenue", "inferred_type": "numeric"},
        ]
    }

    assert date_like_column_names(profile) == [(0, "period"), (1, "period")]


def test_build_period_series_uses_period_column_as_index():
    normalized_df = pd.DataFrame({
        "period": ["2023-01-01", "2023-02-01"],
        "revenue": [100.0, 110.0],
    })

    series = build_period_series(normalized_df, value_position=1, period_position=0)

    assert list(series.index) == ["2023-01-01", "2023-02-01"]
    assert list(series.values) == [100.0, 110.0]


def test_build_period_series_handles_duplicate_column_names_positionally():
    # Both columns are named "col"; position 0 holds period labels, position 1 holds values.
    normalized_df = pd.DataFrame(
        [["2023-01-01", 100.0], ["2023-02-01", 110.0]],
        columns=["col", "col"],
    )

    series = build_period_series(normalized_df, value_position=1, period_position=0)

    assert list(series.index) == ["2023-01-01", "2023-02-01"]
    assert list(series.values) == [100.0, 110.0]


def test_build_monthly_series_aggregates_multiple_rows_in_same_month():
    normalized_df = pd.DataFrame({
        "PurchaseDate": pd.to_datetime(["2023-01-03", "2023-01-15", "2023-02-01"]),
        "TotalPrice": [500.0, 300.0, 700.0],
    })

    series = build_monthly_series(normalized_df, value_position=1, period_position=0)

    assert list(series.index) == ["2023-01", "2023-02"]
    assert list(series.values) == [800.0, 700.0]


def test_build_monthly_series_sorts_chronologically_regardless_of_row_order():
    normalized_df = pd.DataFrame({
        "PurchaseDate": pd.to_datetime(["2024-03-05", "2025-06-21", "2023-06-25"]),
        "TotalPrice": [100.0, 200.0, 50.0],
    })

    series = build_monthly_series(normalized_df, value_position=1, period_position=0)

    assert list(series.index) == ["2023-06", "2024-03", "2025-06"]
    assert list(series.values) == [50.0, 100.0, 200.0]


def test_build_monthly_series_handles_duplicate_column_names_positionally():
    # Both columns are named "col"; position 0 holds dates, position 1 holds values.
    normalized_df = pd.DataFrame(
        [[pd.Timestamp("2023-01-03"), 500.0], [pd.Timestamp("2023-01-15"), 300.0]],
        columns=["col", "col"],
    )

    series = build_monthly_series(normalized_df, value_position=1, period_position=0)

    assert list(series.index) == ["2023-01"]
    assert list(series.values) == [800.0]


def test_build_monthly_series_raises_value_error_for_non_datetime_period_column():
    normalized_df = pd.DataFrame({
        "label": ["not", "a", "date"],
        "amount": [1.0, 2.0, 3.0],
    })

    with pytest.raises(ValueError):
        build_monthly_series(normalized_df, value_position=1, period_position=0)


def test_format_column_option_returns_plain_name_when_unique():
    options = [(0, "revenue"), (1, "count")]

    assert format_column_option((0, "revenue"), options) == "revenue"


def test_format_column_option_disambiguates_duplicate_names_with_position():
    options = [(0, "revenue"), (1, "revenue")]

    assert format_column_option((0, "revenue"), options) == "revenue (col 0)"
    assert format_column_option((1, "revenue"), options) == "revenue (col 1)"


def test_build_analytics_payload_omits_trend_fields_when_not_provided():
    payload = build_analytics_payload({"columns": []}, {"columns": []})

    assert set(payload.keys()) == {"numeric_summary", "date_summary"}


def test_build_analytics_payload_includes_trend_fields_when_provided():
    trend = {"trend": "increasing"}
    period_comparison = {"comparisons": []}

    payload = build_analytics_payload({"columns": []}, {"columns": []}, trend, period_comparison)

    assert payload["trend"] == trend
    assert payload["period_comparison"] == period_comparison


def test_build_analytics_payload_omits_anomalies_when_not_provided():
    payload = build_analytics_payload({"columns": []}, {"columns": []})

    assert "anomalies" not in payload


def test_build_analytics_payload_includes_anomalies_when_provided():
    anomalies = {"method": "iqr", "insufficient_data": False, "anomaly_count": 0, "anomalies": []}

    payload = build_analytics_payload({"columns": []}, {"columns": []}, anomalies=anomalies)

    assert payload["anomalies"] == anomalies


def test_ai_error_message_known_reason():
    assert "GEMINI_API_KEY" in ai_error_message("missing_api_key")


def test_ai_error_message_unknown_reason_has_fallback():
    assert ai_error_message("some_new_unmapped_reason") == "AI insight generation failed."


# --- Anomaly-detection integration with the trend-analysis pipeline -------
#
# These tests reconstruct the exact sequence main() runs inside its Trend
# Analysis branch (build_monthly_series -> detect_trend / compare_periods /
# detect_iqr_anomalies -> build_analytics_payload) so the wiring is verified
# without needing to unit-test main() itself.

def _monthly_normalized_df(dates, values):
    return pd.DataFrame({
        "PurchaseDate": pd.to_datetime(dates),
        "TotalPrice": values,
    })


def test_pipeline_normal_monthly_series_has_no_anomalies():
    normalized_df = _monthly_normalized_df(
        ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01", "2023-06-01"],
        [100.0, 105.0, 98.0, 102.0, 101.0, 99.0],
    )
    series = build_monthly_series(normalized_df, value_position=1, period_position=0)

    anomalies = detect_iqr_anomalies(series)

    assert anomalies["insufficient_data"] is False
    assert anomalies["anomaly_count"] == 0
    assert anomalies["anomalies"] == []


def test_pipeline_monthly_series_with_high_anomaly_is_detected():
    normalized_df = _monthly_normalized_df(
        ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01", "2023-06-01"],
        [100.0, 105.0, 98.0, 102.0, 101.0, 5000.0],
    )
    series = build_monthly_series(normalized_df, value_position=1, period_position=0)

    anomalies = detect_iqr_anomalies(series)

    assert anomalies["anomaly_count"] == 1
    assert anomalies["anomalies"][0]["direction"] == "high"
    assert anomalies["anomalies"][0]["period"] == "2023-06"


def test_pipeline_monthly_series_with_low_anomaly_is_detected():
    normalized_df = _monthly_normalized_df(
        ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01", "2023-06-01"],
        [100.0, 105.0, 98.0, 102.0, 101.0, -5000.0],
    )
    series = build_monthly_series(normalized_df, value_position=1, period_position=0)

    anomalies = detect_iqr_anomalies(series)

    assert anomalies["anomaly_count"] == 1
    assert anomalies["anomalies"][0]["direction"] == "low"
    assert anomalies["anomalies"][0]["period"] == "2023-06"


def test_pipeline_insufficient_monthly_data_reports_insufficient_for_all_three():
    # A single aggregated month: detect_trend/compare_periods already treat
    # this as insufficient data on their own; detect_iqr_anomalies must still
    # run on the same series and independently report its own defined result
    # (insufficient_data=True) rather than being skipped.
    normalized_df = _monthly_normalized_df(["2023-01-01"], [100.0])
    series = build_monthly_series(normalized_df, value_position=1, period_position=0)

    trend = detect_trend(series)
    period_comparison = compare_periods(series)
    anomalies = detect_iqr_anomalies(series)

    assert trend["trend"] == "insufficient_data"
    assert period_comparison["insufficient_data"] is True
    assert anomalies["insufficient_data"] is True
    assert anomalies["anomalies"] == []


def test_pipeline_anomaly_detector_ignores_missing_values_in_a_period_series():
    # build_monthly_series itself cannot produce a None in its aggregated
    # output (groupby().sum() treats an all-missing month as 0.0, not NaN),
    # so this exercises the same detect_iqr_anomalies(series) call the
    # pipeline makes against a period series that does contain a missing
    # value, confirming the integration point tolerates it correctly.
    series = pd.Series([100.0, None, 105.0, 98.0, 102.0, 5000.0], index=[
        "2023-01", "2023-02", "2023-03", "2023-04", "2023-05", "2023-06",
    ])

    anomalies = detect_iqr_anomalies(series)

    assert anomalies["sample_size"] == 5
    assert anomalies["anomaly_count"] == 1
    assert anomalies["anomalies"][0]["period"] == "2023-06"


def test_pipeline_anomalies_present_in_analytics_payload():
    normalized_df = _monthly_normalized_df(
        ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01", "2023-06-01"],
        [100.0, 105.0, 98.0, 102.0, 101.0, 5000.0],
    )
    series = build_monthly_series(normalized_df, value_position=1, period_position=0)
    trend = detect_trend(series)
    period_comparison = compare_periods(series)
    anomalies = detect_iqr_anomalies(series)

    payload = build_analytics_payload({"columns": []}, {"columns": []}, trend, period_comparison, anomalies)

    assert payload["anomalies"] == anomalies
    assert payload["anomalies"]["anomaly_count"] == 1


def test_pipeline_trend_and_period_comparison_unchanged_by_anomaly_detection():
    normalized_df = _monthly_normalized_df(
        ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01", "2023-06-01"],
        [100.0, 105.0, 98.0, 102.0, 101.0, 5000.0],
    )
    series = build_monthly_series(normalized_df, value_position=1, period_position=0)

    trend_before = detect_trend(series)
    period_comparison_before = compare_periods(series)
    detect_iqr_anomalies(series)  # run anomaly detection in between, as main() does
    trend_after = detect_trend(series)
    period_comparison_after = compare_periods(series)

    assert trend_before == trend_after
    assert period_comparison_before == period_comparison_after


def test_pipeline_anomaly_detector_operates_on_the_same_series_not_a_reconstruction():
    normalized_df = _monthly_normalized_df(
        ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01", "2023-06-01"],
        [100.0, 105.0, 98.0, 102.0, 101.0, 5000.0],
    )
    series = build_monthly_series(normalized_df, value_position=1, period_position=0)

    anomalies = detect_iqr_anomalies(series)

    # sample_size must equal the actual non-null length of the same series
    # main() built via build_monthly_series (not an independently
    # re-aggregated dataset), and every reported period label must be one
    # of that series' own index labels.
    assert anomalies["sample_size"] == len(series.dropna())
    assert all(a["period"] in list(series.index) for a in anomalies["anomalies"])
