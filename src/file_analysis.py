"""Per-file deterministic analysis orchestration (V0.7.2).

Pure functions only — no Streamlit dependency. Reuses the existing
analytics/profiling/normalization/anomaly-detection modules unchanged; this
module composes them into one per-file pipeline, `analyze_file()`, so that
app.py can run it once per uploaded file instead of duplicating per-file
analysis logic inline.
"""
import pandas as pd

from src.analytics_engine import analyze_dates, analyze_numeric, compare_periods, detect_trend
from src.anomaly_detector import detect_iqr_anomalies
from src.data_normalizer import normalize_dataframe
from src.data_profiler import profile_dataframe


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


def analyze_file(raw_df: pd.DataFrame, value_position: int | None = None, period_position: int | None = None) -> dict:
    """Run the per-file deterministic analysis pipeline once and return a
    FileAnalysis dict:

        {
            "profile": ...,
            "normalized_df": ...,
            "numeric_summary": ...,
            "date_summary": ...,
            "monthly_series": ...,
            "trend": ...,
            "period_comparison": ...,
            "anomalies": ...,
        }

    Mirrors exactly the sequence app.py's main() previously ran inline for
    its one file: profile -> normalize -> numeric/date summary, then (only
    when both a value and a period column position are given) monthly
    aggregation -> trend -> period comparison -> anomaly detection.

    If the period column can't be aggregated (`build_monthly_series` raises
    ValueError), the trend-dependent fields are left None rather than the
    whole analysis failing — profile/numeric_summary/date_summary are still
    returned, exactly as the previous inline code kept rendering Dataset
    Information / Deterministic Analytics even when Trend Analysis failed.
    """
    profile = profile_dataframe(raw_df)
    normalized_df = normalize_dataframe(raw_df)["normalized_df"]
    numeric_summary = analyze_numeric(normalized_df)
    date_summary = analyze_dates(normalized_df)

    monthly_series = None
    trend = None
    period_comparison = None
    anomalies = None

    if value_position is not None and period_position is not None:
        try:
            monthly_series = build_monthly_series(normalized_df, value_position, period_position)
        except ValueError:
            monthly_series = None
        else:
            trend = detect_trend(monthly_series)
            period_comparison = compare_periods(monthly_series)
            anomalies = detect_iqr_anomalies(monthly_series)

    return {
        "profile": profile,
        "normalized_df": normalized_df,
        "numeric_summary": numeric_summary,
        "date_summary": date_summary,
        "monthly_series": monthly_series,
        "trend": trend,
        "period_comparison": period_comparison,
        "anomalies": anomalies,
    }
