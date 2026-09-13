import pandas as pd
import plotly.graph_objects as go

DEFAULT_TREND_CHART_TITLE = "Monthly Trend"

SUPPORTED_NUMERIC_METRICS = {"sum", "mean", "median", "min", "max", "std", "count"}


def build_trend_chart(monthly_series: pd.Series, title: str | None = None) -> go.Figure:
    """Render an already-aggregated period series (e.g. build_monthly_series's
    output) as a line chart, preserving its index order exactly."""
    fig = go.Figure(
        data=[
            go.Scatter(
                x=list(monthly_series.index),
                y=list(monthly_series.to_numpy()),
                mode="lines+markers",
            )
        ]
    )
    fig.update_layout(
        title=title or DEFAULT_TREND_CHART_TITLE,
        xaxis_title="Period",
        yaxis_title="Value",
    )
    return fig


def build_numeric_summary_chart(numeric_summary: dict, metric: str = "sum") -> go.Figure:
    """Render one analyze_numeric() metric as a bar chart across numeric columns.

    Column order and duplicate names are preserved exactly as they appear in
    numeric_summary["columns"] — no sorting, disambiguation, or recalculation.
    """
    if metric not in SUPPORTED_NUMERIC_METRICS:
        raise ValueError(
            f"Unsupported metric: {metric!r}. Supported metrics: {sorted(SUPPORTED_NUMERIC_METRICS)}."
        )

    columns = numeric_summary["columns"]
    names = [column["name"] for column in columns]
    values = [column[metric] for column in columns]

    fig = go.Figure(data=[go.Bar(x=names, y=values)])
    fig.update_layout(
        title=f"Numeric Summary ({metric})",
        xaxis_title="Column",
        yaxis_title=metric,
    )
    return fig


def build_period_change_chart(period_comparison: dict) -> go.Figure:
    """Render compare_periods()'s absolute_change per transition as a bar chart.

    Comparison order is preserved exactly as provided — no sorting, and
    absolute_change values (including None) are used as-is, with no
    recalculation and no fallback to percentage_change.
    """
    comparisons = period_comparison["comparisons"]
    labels = [f"{comparison['from_period']} → {comparison['to_period']}" for comparison in comparisons]
    values = [comparison["absolute_change"] for comparison in comparisons]

    fig = go.Figure(data=[go.Bar(x=labels, y=values)])
    fig.update_layout(
        title="Period Change",
        xaxis_title="Period",
        yaxis_title="Absolute Change",
    )
    return fig


def build_missing_values_chart(profile: dict) -> go.Figure:
    """Render profile_dataframe()'s missing_percentage per column as a bar chart.

    Column order and duplicate names are preserved exactly as they appear in
    profile["columns"] — no sorting and no recalculation of the percentages.
    """
    columns = profile["columns"]
    names = [column["name"] for column in columns]
    values = [column["missing_percentage"] for column in columns]

    fig = go.Figure(data=[go.Bar(x=names, y=values)])
    fig.update_layout(
        title="Missing Values",
        xaxis_title="Column",
        yaxis_title="Missing Percentage",
    )
    return fig
