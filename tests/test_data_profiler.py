import pandas as pd

from src.data_profiler import profile_dataframe


def test_profile_normal_mixed_dataframe():
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["a", "b", "c"],
    })

    result = profile_dataframe(df)

    assert result["row_count"] == 3
    assert result["column_count"] == 2
    assert result["is_empty"] is False
    assert result["duplicate_row_count"] == 0
    assert result["duplicate_column_names"] == []
    assert len(result["columns"]) == 2


def test_profile_empty_dataframe_with_defined_columns():
    df = pd.DataFrame(columns=["a", "b"])

    result = profile_dataframe(df)

    assert result["row_count"] == 0
    assert result["column_count"] == 2
    assert result["is_empty"] is True
    assert result["duplicate_row_count"] == 0
    for column in result["columns"]:
        assert column["missing_percentage"] == 0.0
        assert column["missing_count"] == 0
        assert column["unique_count"] == 0


def test_profile_missing_values_and_percentage():
    df = pd.DataFrame({"a": [1, None, None, 4]})

    result = profile_dataframe(df)

    column = result["columns"][0]
    assert column["missing_count"] == 2
    assert column["missing_percentage"] == 50.0


def test_profile_duplicate_rows():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})

    result = profile_dataframe(df)

    assert result["duplicate_row_count"] == 1


def test_profile_duplicate_column_names():
    df = pd.DataFrame([[1, 2, 3]], columns=["a", "a", "b"])

    result = profile_dataframe(df)

    assert result["duplicate_column_names"] == ["a"]
    assert result["column_count"] == 3
    assert len(result["columns"]) == 3
    assert [c["name"] for c in result["columns"]] == ["a", "a", "b"]


def test_profile_numeric_classification():
    df = pd.DataFrame({"a": [1, 2, 3]})

    result = profile_dataframe(df)

    assert result["columns"][0]["inferred_type"] == "numeric"


def test_profile_text_classification():
    df = pd.DataFrame({"a": ["apple", "banana", "cherry"]})

    result = profile_dataframe(df)

    assert result["columns"][0]["inferred_type"] == "text"


def test_profile_real_datetime_classification():
    df = pd.DataFrame({"a": pd.to_datetime(["2024-01-01", "2024-01-02"])})

    result = profile_dataframe(df)

    assert result["columns"][0]["inferred_type"] == "date-like"


def test_profile_string_formatted_date_classification():
    df = pd.DataFrame({"a": ["2024-01-01", "2024-02-15", "2024-03-20"]})

    result = profile_dataframe(df)

    assert result["columns"][0]["inferred_type"] == "date-like"
    assert not pd.api.types.is_datetime64_any_dtype(df["a"])


def test_profile_does_not_mutate_original_dataframe():
    original = pd.DataFrame({
        "a": [1, None, 3],
        "b": ["2024-01-01", "x", None],
    })
    before = original.copy(deep=True)

    profile_dataframe(original)

    pd.testing.assert_frame_equal(original, before)
