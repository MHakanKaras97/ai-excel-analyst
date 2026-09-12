import pandas as pd

from app import (
    ai_error_message,
    build_analytics_payload,
    build_period_series,
    date_like_column_names,
    format_column_option,
    numeric_column_names,
    save_uploaded_file,
)


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


def test_ai_error_message_known_reason():
    assert "GEMINI_API_KEY" in ai_error_message("missing_api_key")


def test_ai_error_message_unknown_reason_has_fallback():
    assert ai_error_message("some_new_unmapped_reason") == "AI insight generation failed."
