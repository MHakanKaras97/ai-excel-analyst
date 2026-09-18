import pandas as pd
import pytest

from src.file_analysis import (
    analyze_file,
    build_monthly_series,
    build_period_series,
    date_like_column_names,
    format_column_option,
    numeric_column_names,
)

# ==================================================
# Relocated from tests/test_app_helpers.py (V0.7.2) — moved mechanically,
# behavior unchanged, only the import source changed (app -> src.file_analysis).
# ==================================================


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


# ==================================================
# analyze_file() — new for V0.7.2
# ==================================================


def _raw_df():
    return pd.DataFrame({
        "PurchaseDate": pd.to_datetime([
            "2023-01-03", "2023-01-15", "2023-02-01", "2023-03-01",
            "2023-04-01", "2023-05-01", "2023-06-01",
        ]),
        "TotalPrice": [500.0, 300.0, 700.0, 690.0, 710.0, 705.0, 9000.0],
        "Notes": ["a", "b", "c", "d", "e", "f", "g"],
    })


def test_analyze_file_returns_exact_contract_keys():
    analysis = analyze_file(_raw_df())

    assert set(analysis.keys()) == {
        "profile", "normalized_df", "numeric_summary", "date_summary",
        "monthly_series", "trend", "period_comparison", "anomalies",
    }


def test_analyze_file_without_positions_computes_profile_and_summaries_only():
    analysis = analyze_file(_raw_df())

    assert analysis["profile"]["row_count"] == 7
    assert analysis["profile"]["column_count"] == 3
    assert isinstance(analysis["normalized_df"], pd.DataFrame)
    assert analysis["numeric_summary"]["columns"][0]["name"] == "TotalPrice"
    assert analysis["date_summary"]["columns"][0]["name"] == "PurchaseDate"
    assert analysis["monthly_series"] is None
    assert analysis["trend"] is None
    assert analysis["period_comparison"] is None
    assert analysis["anomalies"] is None


def test_analyze_file_with_positions_computes_trend_dependent_fields():
    analysis = analyze_file(_raw_df(), value_position=1, period_position=0)

    assert analysis["monthly_series"] is not None
    assert list(analysis["monthly_series"].index) == [
        "2023-01", "2023-02", "2023-03", "2023-04", "2023-05", "2023-06",
    ]
    assert analysis["trend"] is not None
    assert analysis["period_comparison"]["comparisons"]
    assert analysis["anomalies"]["anomaly_count"] == 1
    assert analysis["anomalies"]["anomalies"][0]["period"] == "2023-06"


def test_analyze_file_only_one_position_given_leaves_trend_fields_none():
    analysis = analyze_file(_raw_df(), value_position=1, period_position=None)

    assert analysis["monthly_series"] is None
    assert analysis["trend"] is None
    assert analysis["period_comparison"] is None
    assert analysis["anomalies"] is None


def test_analyze_file_handles_unparseable_period_column_without_raising():
    raw_df = pd.DataFrame({
        "label": ["not", "a", "date"],
        "amount": [1.0, 2.0, 3.0],
    })

    analysis = analyze_file(raw_df, value_position=1, period_position=0)

    assert analysis["monthly_series"] is None
    assert analysis["trend"] is None
    assert analysis["period_comparison"] is None
    assert analysis["anomalies"] is None
    # profile/summaries must still be produced even though trend failed.
    assert analysis["profile"]["row_count"] == 3
    assert analysis["numeric_summary"]["columns"][0]["name"] == "amount"


def test_analyze_file_does_not_mutate_input_dataframe():
    raw_df = _raw_df()
    before = raw_df.copy(deep=True)

    analyze_file(raw_df, value_position=1, period_position=0)

    pd.testing.assert_frame_equal(raw_df, before)


def test_analyze_file_reuses_existing_analytics_contracts_verbatim():
    # numeric_summary/date_summary/trend/period_comparison/anomalies must be
    # byte-for-byte the same shapes analytics_engine/anomaly_detector already
    # produce — analyze_file must not reshape or recalculate them.
    from src.analytics_engine import analyze_dates, analyze_numeric, compare_periods, detect_trend
    from src.anomaly_detector import detect_iqr_anomalies
    from src.data_normalizer import normalize_dataframe

    raw_df = _raw_df()
    analysis = analyze_file(raw_df, value_position=1, period_position=0)

    normalized_df = normalize_dataframe(raw_df)["normalized_df"]
    expected_numeric_summary = analyze_numeric(normalized_df)
    expected_date_summary = analyze_dates(normalized_df)
    expected_series = build_monthly_series(normalized_df, 1, 0)
    expected_trend = detect_trend(expected_series)
    expected_period_comparison = compare_periods(expected_series)
    expected_anomalies = detect_iqr_anomalies(expected_series)

    assert analysis["numeric_summary"] == expected_numeric_summary
    assert analysis["date_summary"] == expected_date_summary
    assert analysis["trend"] == expected_trend
    assert analysis["period_comparison"] == expected_period_comparison
    assert analysis["anomalies"] == expected_anomalies
