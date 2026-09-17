"""Renders a qa_chart_engine.build_chart_spec() grounded spec into a Plotly
Figure by dispatching to the existing chart_builder.py functions (V0.6.4).

Purely a thin chart_type -> builder-call mapping: no column/period
resolution, no analytics, no LLM call, and no chart construction logic of
its own — every Figure is built entirely by the existing chart_builder
functions from data already present in the spec.
"""
import pandas as pd
import plotly.graph_objects as go

from src.chart_builder import (
    build_missing_values_chart,
    build_numeric_summary_chart,
    build_period_change_chart,
    build_trend_chart,
)


def _render_trend(spec: dict) -> go.Figure:
    data = spec["data"]
    series = pd.Series(data["values"], index=data["periods"], name=spec["column"])
    return build_trend_chart(series, anomalies=spec["extra"].get("anomalies"))


def _render_numeric_summary(spec: dict) -> go.Figure:
    return build_numeric_summary_chart(spec["data"], metric=spec["metric"])


def _render_period_change(spec: dict) -> go.Figure:
    return build_period_change_chart(spec["data"])


def _render_missing_values(spec: dict) -> go.Figure:
    return build_missing_values_chart(spec["data"])


_RENDERERS = {
    "trend": _render_trend,
    "numeric_summary": _render_numeric_summary,
    "period_change": _render_period_change,
    "missing_values": _render_missing_values,
}


def render_chart_spec(spec: dict) -> go.Figure | None:
    """Render a qa_chart_engine grounded chart spec into a Plotly Figure.

    Returns None for a failed/unsupported spec (spec["found"] is False)
    without ever calling a chart_builder function. Dispatches solely on the
    already-resolved spec["chart_type"] — an unrecognized chart_type raises
    ValueError rather than silently picking a builder.
    """
    if not spec["found"]:
        return None

    chart_type = spec["chart_type"]
    renderer = _RENDERERS.get(chart_type)
    if renderer is None:
        raise ValueError(f"Unsupported chart_type: {chart_type!r}")

    return renderer(spec)
