import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.ai_interpreter import interpret
from src.analytics_engine import analyze_dates, analyze_numeric, compare_periods, detect_trend
from src.data_normalizer import normalize_dataframe
from src.data_profiler import profile_dataframe
from src.excel_loader import load_excel
from src.gemini_provider import GeminiProvider

AI_ERROR_MESSAGES = {
    "missing_api_key": "GEMINI_API_KEY is not set. Add it to your environment to enable AI insights.",
    "provider_unavailable": "The Gemini SDK is not available in this environment.",
    "provider_error": "The Gemini API request failed. Please try again.",
    "empty_response": "Gemini returned an empty response.",
    "malformed_output": "Gemini returned a response that could not be parsed.",
}


def save_uploaded_file(uploaded_file, directory: Path) -> Path:
    """Persist an uploaded Streamlit file to `directory`, preserving its extension."""
    suffix = Path(uploaded_file.name).suffix
    dest = Path(directory) / f"upload{suffix}"
    dest.write_bytes(uploaded_file.getbuffer())
    return dest


def numeric_column_names(df: pd.DataFrame) -> list[tuple[int, str]]:
    """Return (position, name) pairs for numeric columns.

    Positional access avoids the DataFrame-vs-Series ambiguity that label
    indexing (`df[name]`) has when column names are duplicated.
    """
    return [
        (i, str(df.columns[i]))
        for i in range(len(df.columns))
        if pd.api.types.is_numeric_dtype(df.iloc[:, i])
    ]


def date_like_column_names(profile: dict) -> list[tuple[int, str]]:
    """Return (position, name) pairs for date-like columns.

    `profile["columns"]` is ordered by column position (see
    `profile_dataframe`), so each entry's list index is that column's position.
    """
    return [
        (i, c["name"])
        for i, c in enumerate(profile["columns"])
        if c["inferred_type"] == "date-like"
    ]


def format_column_option(option: tuple[int, str], options: list[tuple[int, str]]) -> str:
    """Render a (position, name) option for a selectbox, disambiguating duplicate names."""
    position, name = option
    if sum(1 for _, other_name in options if other_name == name) > 1:
        return f"{name} (col {position})"
    return name


def build_period_series(normalized_df: pd.DataFrame, value_position: int, period_position: int) -> pd.Series:
    return pd.Series(
        normalized_df.iloc[:, value_position].to_numpy(),
        index=normalized_df.iloc[:, period_position].to_numpy(),
    )


def build_monthly_series(normalized_df: pd.DataFrame, value_position: int, period_position: int) -> pd.Series:
    """Aggregate a raw row-level value/period series into monthly sums.

    Reuses `build_period_series` for the positional (duplicate-column-safe)
    extraction, then parses the period axis to datetime, drops rows that
    don't parse, sums by calendar month, and sorts chronologically. Period
    labels are formatted as plain "YYYY-MM" strings so the result stays
    JSON-serializable for `detect_trend`/`compare_periods` without any change
    to analytics_engine.
    """
    raw_series = build_period_series(normalized_df, value_position, period_position)

    period_dt = pd.to_datetime(raw_series.index, errors="coerce")
    if period_dt.notna().sum() == 0:
        raise ValueError("Period column does not contain datetime-compatible values.")

    valid = period_dt.notna()
    monthly = pd.Series(raw_series.to_numpy()[valid], index=period_dt[valid].to_period("M"))
    monthly = monthly.groupby(level=0).sum().sort_index()
    monthly.index = monthly.index.astype(str)
    return monthly


def build_analytics_payload(numeric_summary: dict, date_summary: dict, trend: dict | None = None,
                             period_comparison: dict | None = None) -> dict:
    payload = {"numeric_summary": numeric_summary, "date_summary": date_summary}
    if trend is not None:
        payload["trend"] = trend
    if period_comparison is not None:
        payload["period_comparison"] = period_comparison
    return payload


def ai_error_message(reason: str) -> str:
    return AI_ERROR_MESSAGES.get(reason, "AI insight generation failed.")


def main():
    st.title("AI Excel Analyst")
    st.write(
        "Upload an Excel file to generate deterministic analytics and an "
        "AI-generated summary of the results. Calculations are always "
        "produced by the analytics engine — the AI only interprets them."
    )

    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"], accept_multiple_files=False)

    if uploaded_file is None:
        return

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            excel_path = save_uploaded_file(uploaded_file, Path(tmp_dir))
            df = load_excel(excel_path)
    except Exception as exc:
        st.error(f"Could not read the uploaded file: {exc}")
        return

    profile = profile_dataframe(df)

    st.subheader("Dataset Information")
    st.write(f"Rows: {profile['row_count']} | Columns: {profile['column_count']}")
    if profile["duplicate_row_count"]:
        st.warning(f"{profile['duplicate_row_count']} duplicate row(s) detected.")

    if profile["is_empty"]:
        st.error("The uploaded file contains no rows.")
        return

    st.subheader("Column Information")
    st.dataframe(pd.DataFrame(profile["columns"]))

    normalization_result = normalize_dataframe(df)
    normalized_df = normalization_result["normalized_df"]

    numeric_summary = analyze_numeric(normalized_df)
    date_summary = analyze_dates(normalized_df)

    st.subheader("Deterministic Analytics")
    st.write("Numeric columns")
    if numeric_summary["columns"]:
        st.dataframe(pd.DataFrame(numeric_summary["columns"]))
    else:
        st.info("No numeric columns found.")

    st.write("Date columns")
    if date_summary["columns"]:
        st.dataframe(pd.DataFrame(date_summary["columns"]))
    else:
        st.info("No usable date columns found.")

    trend = None
    period_comparison = None
    value_options = numeric_column_names(normalized_df)
    period_options = date_like_column_names(profile)

    st.subheader("Trend Analysis")
    if not value_options and not period_options:
        st.info("No numeric columns and no usable date columns found; trend analysis is unavailable.")
    elif not value_options:
        st.info("No numeric columns found; trend analysis is unavailable.")
    elif not period_options:
        st.info("No usable date columns found; trend analysis is unavailable.")
    else:
        value_option = st.selectbox(
            "Value column", value_options, format_func=lambda opt: format_column_option(opt, value_options)
        )
        period_option = st.selectbox(
            "Period column", period_options, format_func=lambda opt: format_column_option(opt, period_options)
        )
        st.caption("Values are aggregated by month (sum) before trend analysis.")
        try:
            series = build_monthly_series(normalized_df, value_option[0], period_option[0])
        except ValueError as exc:
            st.error(f"Could not aggregate the selected period column: {exc}")
        else:
            trend = detect_trend(series)
            period_comparison = compare_periods(series)
            st.write(f"Trend: {trend['trend']}")
            if period_comparison["comparisons"]:
                st.dataframe(pd.DataFrame(period_comparison["comparisons"]))

    analytics_payload = build_analytics_payload(numeric_summary, date_summary, trend, period_comparison)

    st.subheader("AI-Generated Insight")

    if not os.environ.get("GEMINI_API_KEY"):
        st.error(ai_error_message("missing_api_key"))
        return

    cache_key = json.dumps(analytics_payload, sort_keys=True, default=str)

    if st.button("Generate AI Insight"):
        provider = GeminiProvider()
        st.session_state["ai_result"] = interpret(analytics_payload, provider)
        st.session_state["ai_cache_key"] = cache_key

    if st.session_state.get("ai_cache_key") == cache_key:
        result = st.session_state.get("ai_result")
        if result is not None:
            if result["is_valid"]:
                insight = result["insight"]
                st.write(insight["summary"])
                if insight["key_insights"]:
                    st.write("**Key insights**")
                    for item in insight["key_insights"]:
                        st.write(f"- {item}")
                if insight["trend_interpretation"]:
                    st.write(f"**Trend interpretation:** {insight['trend_interpretation']}")
                if insight["warnings"]:
                    st.write("**Warnings**")
                    for item in insight["warnings"]:
                        st.write(f"- {item}")
                if insight["recommendations"]:
                    st.write("**Recommendations**")
                    for item in insight["recommendations"]:
                        st.write(f"- {item}")
            else:
                st.error(ai_error_message(result["reason"]))


if __name__ == "__main__":
    main()
