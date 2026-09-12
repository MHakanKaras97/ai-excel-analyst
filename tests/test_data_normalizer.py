import math

import pandas as pd

from src.data_normalizer import normalize_dataframe, normalize_series, normalize_value


def test_numeric_passthrough():
    result = normalize_value(42)
    assert result["normalized_value"] == 42
    assert result["normalized_type"] == "numeric"
    assert result["was_changed"] is False
    assert result["is_ambiguous"] is False

    result_float = normalize_value(3.14)
    assert result_float["normalized_type"] == "numeric"
    assert result_float["was_changed"] is False


def test_plain_text_passthrough():
    result = normalize_value("hello world")

    assert result["normalized_value"] == "hello world"
    assert result["normalized_type"] is None
    assert result["was_changed"] is False
    assert result["is_ambiguous"] is False


def test_none_passthrough():
    result = normalize_value(None)

    assert result["normalized_value"] is None
    assert result["normalized_type"] is None
    assert result["is_ambiguous"] is False


def test_nan_passthrough():
    result = normalize_value(float("nan"))

    assert math.isnan(result["normalized_value"])
    assert result["normalized_type"] is None
    assert result["is_ambiguous"] is False


def test_simple_numeric_string():
    result = normalize_value("123.45")

    assert result["normalized_value"] == 123.45
    assert result["normalized_type"] == "numeric"
    assert result["was_changed"] is True
    assert result["is_ambiguous"] is False


def test_comma_thousands():
    result = normalize_value("1,234.56")

    assert result["normalized_value"] == 1234.56
    assert result["normalized_type"] == "numeric"
    assert result["was_changed"] is True


def test_malformed_comma_grouping():
    result = normalize_value("1,23")

    assert result["is_ambiguous"] is True
    assert result["normalized_type"] is None
    assert result["normalized_value"] == "1,23"
    assert result["warning"] == "ambiguous_thousands_separator"


def test_currency_prefix():
    result = normalize_value("$1,200")

    assert result["normalized_value"] == 1200.0
    assert result["normalized_type"] == "currency"
    assert result["was_changed"] is True
    assert result["is_ambiguous"] is False


def test_currency_suffix():
    result = normalize_value("1200€")

    assert result["normalized_value"] == 1200.0
    assert result["normalized_type"] == "currency"
    assert result["was_changed"] is True


def test_unrecognized_and_multiple_currency_symbols():
    unrecognized = normalize_value("₹100")
    assert unrecognized["is_ambiguous"] is True
    assert unrecognized["normalized_value"] == "₹100"

    multiple = normalize_value("$100$")
    assert multiple["is_ambiguous"] is True
    assert multiple["warning"] == "multiple_currency_symbols"


def test_percentage():
    result = normalize_value("25%")

    assert result["normalized_value"] == 0.25
    assert result["normalized_type"] == "percentage"
    assert result["was_changed"] is True
    assert result["is_ambiguous"] is False


def test_malformed_percentage():
    result = normalize_value("25%%")

    assert result["is_ambiguous"] is True
    assert result["normalized_type"] is None
    assert result["normalized_value"] == "25%%"


def test_iso_date():
    result = normalize_value("2024-03-05")

    assert result["normalized_value"] == "2024-03-05"
    assert result["normalized_type"] == "date"
    assert result["was_changed"] is False
    assert result["is_ambiguous"] is False


def test_named_month_date():
    result = normalize_value("March 5, 2024")

    assert result["normalized_value"] == "2024-03-05"
    assert result["normalized_type"] == "date"
    assert result["was_changed"] is True

    result_reversed = normalize_value("5 March 2024")
    assert result_reversed["normalized_value"] == "2024-03-05"
    assert result_reversed["normalized_type"] == "date"
    assert result_reversed["was_changed"] is True


def test_ambiguous_numeric_date():
    result = normalize_value("01/02/2024")

    assert result["is_ambiguous"] is True
    assert result["normalized_type"] is None
    assert result["warning"] == "ambiguous_date_format"


def test_ambiguous_single_dot_three_digits():
    result = normalize_value("1.234")

    assert result["is_ambiguous"] is True
    assert result["normalized_type"] is None
    assert result["normalized_value"] == "1.234"


def test_negative_numeric_value():
    result_native = normalize_value(-42)
    assert result_native["normalized_value"] == -42
    assert result_native["normalized_type"] == "numeric"

    result_string = normalize_value("-42.5")
    assert result_string["normalized_value"] == -42.5
    assert result_string["normalized_type"] == "numeric"
    assert result_string["was_changed"] is True


def test_original_value_preserved_for_unresolved_input():
    raw = "1,23"
    result = normalize_value(raw)

    assert result["raw"] == raw
    assert result["normalized_value"] == raw
    assert result["is_ambiguous"] is True


def test_was_changed_reflects_actual_value_difference():
    numeric = normalize_value("123")
    assert numeric["normalized_value"] == 123.0
    assert numeric["was_changed"] is True

    currency = normalize_value("$1,200")
    assert currency["normalized_value"] == 1200.0
    assert currency["was_changed"] is True

    percentage = normalize_value("25%")
    assert percentage["normalized_value"] == 0.25
    assert percentage["was_changed"] is True

    slash_date = normalize_value("2024/03/05")
    assert slash_date["normalized_value"] == "2024-03-05"
    assert slash_date["normalized_type"] == "date"
    assert slash_date["was_changed"] is True

    unchanged_iso_date = normalize_value("2024-03-05")
    assert unchanged_iso_date["normalized_value"] == "2024-03-05"
    assert unchanged_iso_date["normalized_type"] == "date"
    assert unchanged_iso_date["was_changed"] is False


def test_normalize_series_mixed_values():
    series = pd.Series(["123.45", "$1,200", "1,23", None, "hello"])

    result = normalize_series(series)
    report = result["report"]
    values = list(result["normalized_series"])

    assert values[0] == 123.45
    assert values[1] == 1200.0
    assert values[2] == "1,23"
    assert pd.isna(values[3])
    assert values[4] == "hello"
    assert report["total"] == 5
    assert report["changed_count"] == 2
    assert report["unresolved_count"] == 1
    assert report["missing_count"] == 1


def test_normalize_series_numeric_passthrough():
    series = pd.Series([1, 2, 3])

    result = normalize_series(series)
    report = result["report"]

    pd.testing.assert_series_equal(result["normalized_series"], series)
    assert report["changed_count"] == 0
    assert report["unresolved_count"] == 0
    assert report["missing_count"] == 0


def test_normalize_series_datetime_passthrough():
    series = pd.to_datetime(pd.Series(["2024-01-01", "2024-02-01"]))

    result = normalize_series(series)
    report = result["report"]

    pd.testing.assert_series_equal(result["normalized_series"], series)
    assert report["changed_count"] == 0
    assert report["unresolved_count"] == 0


def test_normalize_series_empty():
    series = pd.Series([], dtype=object)

    result = normalize_series(series)
    report = result["report"]

    assert len(result["normalized_series"]) == 0
    assert report["total"] == 0
    assert report["changed_count"] == 0
    assert report["unresolved_count"] == 0
    assert report["missing_count"] == 0
    assert report["examples"]["unresolved"] == []


def test_normalize_series_preserves_index():
    series = pd.Series(["123", "abc", "1,23"], index=["x", "y", "z"])

    result = normalize_series(series)

    assert list(result["normalized_series"].index) == ["x", "y", "z"]


def test_normalize_series_unresolved_example_reporting():
    series = pd.Series(["1,23"])

    result = normalize_series(series)
    examples = result["report"]["examples"]["unresolved"]

    assert len(examples) == 1
    assert examples[0]["raw"] == "1,23"
    assert examples[0]["warning"] == "ambiguous_thousands_separator"


def test_normalize_series_unresolved_example_limit_of_five():
    ambiguous_values = [f"1,2{i}" for i in range(7)]
    series = pd.Series(ambiguous_values)

    result = normalize_series(series)
    report = result["report"]

    assert report["unresolved_count"] == 7
    assert len(report["examples"]["unresolved"]) == 5


def test_normalize_series_does_not_mutate_original():
    series = pd.Series(["123", "abc", None])
    before = series.copy(deep=True)

    normalize_series(series)

    pd.testing.assert_series_equal(series, before)


def test_normalize_dataframe_multiple_columns_different_behavior():
    df = pd.DataFrame({
        "amount": ["123.45", "$1,200", "1,23"],
        "count": [1, 2, 3],
        "label": ["a", "b", "c"],
    })

    result = normalize_dataframe(df)
    report = result["report"]

    assert report["total_columns"] == 3
    assert [c["name"] for c in report["columns"]] == ["amount", "count", "label"]
    assert report["columns"][0]["changed_count"] == 2
    assert report["columns"][0]["unresolved_count"] == 1
    assert report["columns"][1]["changed_count"] == 0
    assert report["columns"][2]["changed_count"] == 0


def test_normalize_dataframe_numeric_column_passthrough():
    df = pd.DataFrame({"a": [1, 2, 3]})

    result = normalize_dataframe(df)

    pd.testing.assert_series_equal(result["normalized_df"]["a"], df["a"])
    assert result["report"]["columns"][0]["changed_count"] == 0
    assert result["report"]["columns"][0]["unresolved_count"] == 0


def test_normalize_dataframe_mixed_string_numeric():
    df = pd.DataFrame({"a": ["10", "$20", "hello"]})

    result = normalize_dataframe(df)
    values = list(result["normalized_df"]["a"])

    assert values == [10.0, 20.0, "hello"]


def test_normalize_dataframe_date_string_normalization():
    df = pd.DataFrame({"a": ["2024-01-01", "March 5, 2024"]})

    result = normalize_dataframe(df)
    values = list(result["normalized_df"]["a"])

    assert values == ["2024-01-01", "2024-03-05"]
    assert result["report"]["columns"][0]["changed_count"] == 1


def test_normalize_dataframe_missing_values():
    df = pd.DataFrame({"a": ["123", None, "456"]})

    result = normalize_dataframe(df)
    report = result["report"]["columns"][0]

    assert report["missing_count"] == 1
    assert pd.isna(list(result["normalized_df"]["a"])[1])


def test_normalize_dataframe_unresolved_report_propagation():
    df = pd.DataFrame({"a": ["1,23", "01/02/2024"]})

    result = normalize_dataframe(df)
    report = result["report"]["columns"][0]

    assert report["unresolved_count"] == 2
    assert len(report["examples"]["unresolved"]) == 2
    assert report["examples"]["unresolved"][0]["raw"] == "1,23"


def test_normalize_dataframe_empty():
    df = pd.DataFrame()

    result = normalize_dataframe(df)

    assert result["report"]["total_columns"] == 0
    assert result["report"]["columns"] == []
    assert list(result["normalized_df"].columns) == []


def test_normalize_dataframe_duplicate_column_names():
    df = pd.DataFrame([["10", "20"]], columns=["a", "a"])

    result = normalize_dataframe(df)

    assert list(result["normalized_df"].columns) == ["a", "a"]
    assert len(result["report"]["columns"]) == 2
    assert result["report"]["columns"][0]["position"] == 0
    assert result["report"]["columns"][1]["position"] == 1
    assert list(result["normalized_df"].iloc[:, 0]) == [10.0]
    assert list(result["normalized_df"].iloc[:, 1]) == [20.0]


def test_normalize_dataframe_does_not_mutate_original():
    df = pd.DataFrame({"a": ["123", "abc"], "b": [1, 2]})
    before = df.copy(deep=True)

    normalize_dataframe(df)

    pd.testing.assert_frame_equal(df, before)


def test_normalize_dataframe_preserves_index():
    df = pd.DataFrame({"a": ["123", "456"]}, index=["x", "y"])

    result = normalize_dataframe(df)

    assert list(result["normalized_df"].index) == ["x", "y"]


def test_normalize_dataframe_preserves_column_order_and_names():
    df = pd.DataFrame({"z": [1], "a": [2], "m": [3]})

    result = normalize_dataframe(df)

    assert list(result["normalized_df"].columns) == ["z", "a", "m"]
    assert [c["name"] for c in result["report"]["columns"]] == ["z", "a", "m"]
