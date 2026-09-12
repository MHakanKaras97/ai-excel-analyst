import json
from datetime import date

import numpy as np
import pandas as pd

from src.analytics_engine import (
    analyze_dates,
    analyze_numeric,
    compare_periods,
    compare_values,
    detect_trend,
)


def test_analyze_numeric_normal_mixed_dataframe():
    df = pd.DataFrame({"amount": [10, 20, 30], "label": ["a", "b", "c"]})

    result = analyze_numeric(df)

    assert len(result["columns"]) == 1
    col = result["columns"][0]
    assert col["name"] == "amount"
    assert col["count"] == 3
    assert col["missing_count"] == 0
    assert col["sum"] == 60.0
    assert col["mean"] == 20.0
    assert col["median"] == 20.0
    assert col["min"] == 10.0
    assert col["max"] == 30.0
    assert col["std"] == 10.0


def test_analyze_functions_on_empty_dataframe():
    df = pd.DataFrame(columns=["a", "b"])

    assert analyze_numeric(df) == {"columns": []}
    assert analyze_dates(df) == {"columns": []}


def test_analyze_numeric_no_numeric_columns():
    df = pd.DataFrame({"label": ["a", "b", "c"]})

    result = analyze_numeric(df)

    assert result == {"columns": []}


def test_analyze_numeric_missing_values():
    df = pd.DataFrame({"a": [1, None, None, 4]})

    col = analyze_numeric(df)["columns"][0]

    assert col["count"] == 2
    assert col["missing_count"] == 2
    assert col["sum"] == 5.0
    assert col["mean"] == 2.5


def test_analyze_numeric_all_missing_column():
    df = pd.DataFrame({"a": pd.Series([None, None, None], dtype="float64")})

    col = analyze_numeric(df)["columns"][0]

    assert col["count"] == 0
    assert col["missing_count"] == 3
    for key in ("sum", "mean", "median", "min", "max", "std"):
        assert col[key] is None


def test_analyze_numeric_single_row_std_is_none():
    df = pd.DataFrame({"a": [42]})

    col = analyze_numeric(df)["columns"][0]

    assert col["count"] == 1
    assert col["std"] is None
    assert col["mean"] == 42.0


def test_analyze_numeric_constant_column_std_is_zero():
    df = pd.DataFrame({"a": [5, 5, 5]})

    col = analyze_numeric(df)["columns"][0]

    assert col["std"] == 0.0
    assert col["min"] == col["max"] == 5.0


def test_analyze_numeric_negative_and_zero_values():
    df = pd.DataFrame({"a": [-10, 0, 10]})

    col = analyze_numeric(df)["columns"][0]

    assert col["sum"] == 0.0
    assert col["min"] == -10.0
    assert col["max"] == 10.0


def test_analyze_numeric_duplicate_column_names():
    df = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "a"])

    result = analyze_numeric(df)

    assert len(result["columns"]) == 2
    assert result["columns"][0]["name"] == "a"
    assert result["columns"][0]["sum"] == 4.0
    assert result["columns"][1]["sum"] == 6.0


def test_compare_values_normal_case():
    result = compare_values(100000, 125000)

    assert result["absolute_change"] == 25000
    assert result["percentage_change"] == 25.0
    assert result["is_valid"] is True
    assert result["reason"] is None


def test_compare_values_division_by_zero():
    result = compare_values(0, 50)

    assert result["absolute_change"] == 50
    assert result["percentage_change"] is None
    assert result["is_valid"] is False
    assert result["reason"] == "division_by_zero"


def test_compare_values_zero_to_zero():
    result = compare_values(0, 0)

    assert result["absolute_change"] == 0
    assert result["percentage_change"] == 0.0
    assert result["is_valid"] is True
    assert result["reason"] is None


def test_compare_values_missing_input():
    result = compare_values(None, 50)

    assert result["previous"] is None
    assert result["absolute_change"] is None
    assert result["percentage_change"] is None
    assert result["is_valid"] is False
    assert result["reason"] == "missing_value"

    nan_result = compare_values(float("nan"), 50)
    assert nan_result["reason"] == "missing_value"


def test_analyze_dates_real_datetime_column():
    df = pd.DataFrame({"a": pd.to_datetime(["2023-01-01", "2023-12-31"])})

    col = analyze_dates(df)["columns"][0]

    assert col["count"] == 2
    assert col["missing_count"] == 0
    assert col["min"] == pd.Timestamp("2023-01-01").isoformat()
    assert col["max"] == pd.Timestamp("2023-12-31").isoformat()
    assert col["duration_days"] == 364


def test_analyze_dates_string_formatted_column_excluded():
    df = pd.DataFrame({"a": ["2023-01-01", "2023-12-31"]})

    result = analyze_dates(df)

    assert result == {"columns": []}


def test_analyze_dates_empty_column():
    df = pd.DataFrame({"a": pd.to_datetime([None, None])})

    col = analyze_dates(df)["columns"][0]

    assert col["count"] == 0
    assert col["missing_count"] == 2
    assert col["min"] is None
    assert col["max"] is None
    assert col["duration_days"] is None


def test_analyze_functions_do_not_mutate_original_dataframe():
    original = pd.DataFrame({
        "a": [1, None, 3],
        "b": pd.to_datetime(["2023-01-01", None, "2023-03-01"]),
    })
    before = original.copy(deep=True)

    analyze_numeric(original)
    analyze_dates(original)

    pd.testing.assert_frame_equal(original, before)


def test_compare_periods_normal_increasing_sequence():
    values = pd.Series([100, 110, 121], index=["Q1", "Q2", "Q3"])

    result = compare_periods(values)

    assert result["periods"] == ["Q1", "Q2", "Q3"]
    assert result["insufficient_data"] is False
    assert len(result["comparisons"]) == 2
    assert result["comparisons"][0]["from_period"] == "Q1"
    assert result["comparisons"][0]["to_period"] == "Q2"
    assert result["comparisons"][0]["absolute_change"] == 10
    assert result["valid_comparison_count"] == 2
    assert result["invalid_comparison_count"] == 0


def test_compare_periods_empty_series():
    values = pd.Series([], dtype="float64")

    result = compare_periods(values)

    assert result["insufficient_data"] is True
    assert result["comparisons"] == []
    assert result["valid_comparison_count"] == 0
    assert result["invalid_comparison_count"] == 0


def test_compare_periods_single_value_series():
    values = pd.Series([100], index=["Q1"])

    result = compare_periods(values)

    assert result["insufficient_data"] is True
    assert result["comparisons"] == []


def test_compare_periods_all_missing_values():
    values = pd.Series([None, None, None], index=["Q1", "Q2", "Q3"], dtype="float64")

    result = compare_periods(values)

    assert result["insufficient_data"] is False
    assert result["valid_comparison_count"] == 0
    assert result["invalid_comparison_count"] == 2
    assert all(c["reason"] == "missing_value" for c in result["comparisons"])


def test_compare_periods_interspersed_missing_values():
    values = pd.Series([100, None, 120], index=["Q1", "Q2", "Q3"])

    result = compare_periods(values)

    assert result["comparisons"][0]["is_valid"] is False
    assert result["comparisons"][0]["reason"] == "missing_value"
    assert result["comparisons"][1]["is_valid"] is False
    assert result["comparisons"][1]["reason"] == "missing_value"
    assert result["valid_comparison_count"] == 0
    assert result["invalid_comparison_count"] == 2


def test_compare_periods_zero_crossing_division_by_zero():
    values = pd.Series([0, 50], index=["Q1", "Q2"])

    result = compare_periods(values)

    comparison = result["comparisons"][0]
    assert comparison["is_valid"] is False
    assert comparison["reason"] == "division_by_zero"
    assert comparison["absolute_change"] == 50
    assert comparison["percentage_change"] is None
    assert result["valid_comparison_count"] == 0
    assert result["invalid_comparison_count"] == 1


def test_compare_periods_duplicate_period_labels():
    values = pd.Series([100, 150, 200], index=["Q1", "Q1", "Q1"])

    result = compare_periods(values)

    assert len(result["comparisons"]) == 2
    assert result["comparisons"][0]["absolute_change"] == 50
    assert result["comparisons"][1]["absolute_change"] == 50


def test_compare_periods_non_numeric_dtype():
    values = pd.Series(["a", "b", "c"], index=["Q1", "Q2", "Q3"])

    result = compare_periods(values)

    assert result["insufficient_data"] is True
    assert result["comparisons"] == []


def test_compare_periods_timestamp_index_labels_are_json_safe():
    values = pd.Series(
        [100, 110, 120],
        index=pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"]),
    )

    result = compare_periods(values)

    assert result["periods"] == ["2023-01-01T00:00:00", "2023-02-01T00:00:00", "2023-03-01T00:00:00"]
    assert all(isinstance(p, str) for p in result["periods"])
    assert result["comparisons"][0]["from_period"] == "2023-01-01T00:00:00"
    assert result["comparisons"][0]["to_period"] == "2023-02-01T00:00:00"
    json.dumps(result)  # must not raise


def test_compare_periods_python_date_index_labels_are_json_safe():
    values = pd.Series(
        [100, 110, 120],
        index=[date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1)],
    )

    result = compare_periods(values)

    assert result["periods"] == ["2023-01-01T00:00:00", "2023-02-01T00:00:00", "2023-03-01T00:00:00"]
    json.dumps(result)  # must not raise


def test_compare_periods_numpy_datetime64_index_labels_are_json_safe():
    values = pd.Series(
        [100, 110],
        index=pd.Index([np.datetime64("2023-01-01"), np.datetime64("2023-02-01")]),
    )

    result = compare_periods(values)

    assert result["periods"] == ["2023-01-01T00:00:00", "2023-02-01T00:00:00"]
    json.dumps(result)  # must not raise


def test_to_native_still_handles_strings_ints_floats_numpy_and_missing():
    values = pd.Series([1, 2], index=["a", "b"])

    result = compare_periods(values)

    assert result["periods"] == ["a", "b"]
    assert result["comparisons"][0]["previous"] == 1
    assert result["comparisons"][0]["current"] == 2
    assert isinstance(result["comparisons"][0]["previous"], int)

    missing_result = compare_values(None, 5.5)
    assert missing_result["previous"] is None
    assert missing_result["current"] == 5.5
    assert isinstance(missing_result["current"], float)


def test_compare_periods_does_not_mutate_original_series():
    values = pd.Series([100, None, 120], index=["Q1", "Q2", "Q3"])
    before = values.copy(deep=True)

    compare_periods(values)

    pd.testing.assert_series_equal(values, before)


def test_detect_trend_increasing():
    values = pd.Series([100, 110, 130])

    result = detect_trend(values)

    assert result["trend"] == "increasing"
    assert result["increase_count"] == 2
    assert result["decrease_count"] == 0
    assert result["direction_comparison_count"] == 2


def test_detect_trend_decreasing():
    values = pd.Series([130, 110, 100])

    result = detect_trend(values)

    assert result["trend"] == "decreasing"
    assert result["decrease_count"] == 2


def test_detect_trend_stable():
    values = pd.Series([100, 100, 100])

    result = detect_trend(values)

    assert result["trend"] == "stable"
    assert result["no_change_count"] == 2


def test_detect_trend_volatile():
    values = pd.Series([100, 130, 90, 150])

    result = detect_trend(values)

    assert result["trend"] == "volatile"
    assert result["increase_count"] == 2
    assert result["decrease_count"] == 1


def test_detect_trend_insufficient_data_cases():
    empty_result = detect_trend(pd.Series([], dtype="float64"))
    assert empty_result["trend"] == "insufficient_data"
    assert empty_result["direction_comparison_count"] == 0

    single_result = detect_trend(pd.Series([100]))
    assert single_result["trend"] == "insufficient_data"

    all_missing_result = detect_trend(pd.Series([None, None], dtype="float64"))
    assert all_missing_result["trend"] == "insufficient_data"
    assert all_missing_result["direction_comparison_count"] == 0
    assert all_missing_result["invalid_comparison_count"] == 1


def test_detect_trend_direction_aware_division_by_zero():
    values = pd.Series([0, 50, 100])

    result = detect_trend(values)

    assert result["trend"] == "increasing"
    assert result["direction_comparison_count"] == 2
    assert result["valid_comparison_count"] == 1
    assert result["invalid_comparison_count"] == 1


def test_detect_trend_does_not_mutate_original_series():
    values = pd.Series([100, 110, 120])
    before = values.copy(deep=True)

    detect_trend(values)

    pd.testing.assert_series_equal(values, before)
