import pandas as pd

from src.analytics_engine import analyze_dates, analyze_numeric, compare_values


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
