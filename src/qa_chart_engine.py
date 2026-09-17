"""Deterministic chart intent dispatch (V0.6.3).

Takes the structured Chart Intent produced by chart_intent_interpreter.py and
resolves/validates it against already-computed analysis data, producing a
JSON-safe grounded chart specification. No LLM call, no Plotly, no new
analytics — column/period resolution is delegated entirely to
qa_engine.resolve_column()/resolve_period() rather than reimplemented here.
"""
import copy

import pandas as pd

from src.qa_engine import resolve_column, resolve_period

NUMERIC_SUMMARY_METRICS = {"sum", "mean", "median", "min", "max", "std", "count"}
DEFAULT_NUMERIC_SUMMARY_METRIC = "sum"


def _json_safe(value):
    """Convert a pandas/numpy scalar to a plain JSON-serializable value."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def _spec(found: bool, reason, intent_name: str, chart_type=None, column=None, metric=None,
          period=None, from_period=None, to_period=None, data=None, extra=None) -> dict:
    return {
        "found": found,
        "reason": reason,
        "intent": intent_name,
        "chart_type": chart_type,
        "column": column,
        "metric": metric,
        "period": period,
        "from_period": from_period,
        "to_period": to_period,
        "data": data or {},
        "extra": extra or {},
    }


def _failure(reason: str, intent_name: str, extra=None) -> dict:
    return _spec(False, reason, intent_name, extra=extra)


def _success(intent_name: str, chart_type: str, column=None, metric=None, period=None,
             from_period=None, to_period=None, data=None, extra=None) -> dict:
    return _spec(True, None, intent_name, chart_type=chart_type, column=column, metric=metric,
                 period=period, from_period=from_period, to_period=to_period, data=data, extra=extra)


def _numeric_summary_columns(analysis_payload: dict) -> list:
    return (analysis_payload.get("numeric_summary") or {}).get("columns") or []


def _resolve_trend_column(column_hint, analysis_payload: dict, monthly_series):
    """Resolve an optional column hint against known numeric columns, guarding
    against it silently pointing at a different series than the already
    computed `monthly_series` (see qa_engine._resolve_trend_column for the
    identical rule applied to Q&A dispatch).

    Returns (ok, column_name_or_None, reason_or_None, candidates).
    """
    series_name = getattr(monthly_series, "name", None) if monthly_series is not None else None

    if not column_hint:
        return True, series_name, None, []

    column_names = [c["name"] for c in _numeric_summary_columns(analysis_payload)]
    result = resolve_column(column_hint, column_names)
    if not result["found"]:
        return False, None, result["reason"], result["candidates"]

    resolved_column = result["column"]
    if series_name is not None and resolved_column != series_name:
        return False, None, "column_series_mismatch", []

    return True, resolved_column, None, []


def _dispatch_trend_chart(intent: dict, analysis_payload: dict, monthly_series, anomalies) -> dict:
    if monthly_series is None or len(monthly_series) == 0:
        return _failure("trend_not_computed", "trend_chart")

    ok, column, reason, candidates = _resolve_trend_column(intent.get("column_hint"), analysis_payload, monthly_series)
    if not ok:
        return _failure(reason, "trend_chart", extra={"candidates": candidates} if candidates else None)

    data = {
        "periods": [_json_safe(label) for label in monthly_series.index],
        "values": [_json_safe(value) for value in monthly_series.to_numpy()],
    }
    return _success("trend_chart", "trend", column=column, data=data, extra={"anomalies": anomalies})


def _dispatch_numeric_summary_chart(intent: dict, analysis_payload: dict, monthly_series, anomalies) -> dict:
    column_hint = intent.get("column_hint")
    if not column_hint:
        return _failure("column_required", "numeric_summary_chart")

    column_names = [c["name"] for c in _numeric_summary_columns(analysis_payload)]
    result = resolve_column(column_hint, column_names)
    if not result["found"]:
        candidates = result["candidates"]
        return _failure(result["reason"], "numeric_summary_chart", extra={"candidates": candidates} if candidates else None)

    metric = intent.get("metric")
    if metric is None:
        metric = DEFAULT_NUMERIC_SUMMARY_METRIC
    elif metric not in NUMERIC_SUMMARY_METRICS:
        return _failure("unsupported_metric", "numeric_summary_chart")

    column = result["column"]
    column_entry = next((c for c in _numeric_summary_columns(analysis_payload) if c["name"] == column), None)
    if column_entry is None:
        return _failure("column_not_found", "numeric_summary_chart")

    filtered_summary = {"columns": [dict(column_entry)]}
    return _success("numeric_summary_chart", "numeric_summary", column=column, metric=metric, data=filtered_summary)


def _dispatch_period_change_chart(intent: dict, analysis_payload: dict, monthly_series, anomalies) -> dict:
    period_comparison = analysis_payload.get("period_comparison")
    if not period_comparison:
        return _failure("period_comparison_not_computed", "period_change_chart")

    ok, column, reason, candidates = _resolve_trend_column(intent.get("column_hint"), analysis_payload, monthly_series)
    if not ok:
        return _failure(reason, "period_change_chart", extra={"candidates": candidates} if candidates else None)

    from_hint = intent.get("from_period_hint")
    to_hint = intent.get("to_period_hint")

    if from_hint and not to_hint:
        return _failure("period_range_incomplete", "period_change_chart")
    if to_hint and not from_hint:
        return _failure("period_range_incomplete", "period_change_chart")

    if not from_hint and not to_hint:
        return _success("period_change_chart", "period_change", column=column, data=copy.deepcopy(period_comparison))

    known_periods = list(period_comparison.get("periods") or [])

    from_result = resolve_period(from_hint, known_periods)
    if not from_result["found"]:
        candidates = from_result["candidates"]
        return _failure(from_result["reason"], "period_change_chart", extra={"candidates": candidates} if candidates else None)

    to_result = resolve_period(to_hint, known_periods)
    if not to_result["found"]:
        candidates = to_result["candidates"]
        return _failure(to_result["reason"], "period_change_chart", extra={"candidates": candidates} if candidates else None)

    from_period = from_result["period"]
    to_period = to_result["period"]

    matched = next(
        (
            c for c in period_comparison.get("comparisons") or []
            if c.get("from_period") == from_period and c.get("to_period") == to_period
        ),
        None,
    )
    if matched is None:
        return _failure("period_change_not_found", "period_change_chart")

    filtered_comparisons = [dict(matched)]
    filtered_period_comparison = {
        "periods": list(period_comparison.get("periods") or []),
        "comparisons": filtered_comparisons,
        "valid_comparison_count": sum(1 for c in filtered_comparisons if c.get("is_valid")),
        "invalid_comparison_count": sum(1 for c in filtered_comparisons if not c.get("is_valid")),
        "insufficient_data": False,
    }
    return _success(
        "period_change_chart", "period_change", column=column,
        from_period=from_period, to_period=to_period, data=filtered_period_comparison,
    )


def _dispatch_missing_values_chart(intent: dict, analysis_payload: dict, monthly_series, anomalies) -> dict:
    profile = analysis_payload.get("profile")
    if not profile:
        return _failure("profile_not_available", "missing_values_chart")

    return _success("missing_values_chart", "missing_values", data=profile)


def _dispatch_unsupported(intent: dict, analysis_payload: dict, monthly_series, anomalies) -> dict:
    return _failure("unsupported", "unsupported")


_DISPATCHERS = {
    "trend_chart": _dispatch_trend_chart,
    "numeric_summary_chart": _dispatch_numeric_summary_chart,
    "period_change_chart": _dispatch_period_change_chart,
    "missing_values_chart": _dispatch_missing_values_chart,
    "unsupported": _dispatch_unsupported,
}


def build_chart_spec(intent: dict, analysis_payload: dict, monthly_series: pd.Series | None = None,
                      anomalies: dict | None = None) -> dict:
    """Dispatch a validated structured Chart Intent against already-computed
    analysis data and return a grounded, JSON-safe chart specification.

    Purely deterministic: no LLM call, no Plotly, no new analytics/anomaly
    calculation, no mutation of any input. `analysis_payload` is expected to
    hold whichever of "numeric_summary", "period_comparison", "profile" are
    relevant to the intent, exactly as produced by the existing analytics/
    profiling modules. `monthly_series` is the already-computed trend Series
    for series-based intents; `anomalies` is the already-computed anomaly
    result, passed through untouched for trend_chart to hand to
    chart_builder.build_trend_chart() in a later milestone.
    """
    intent_name = intent.get("intent")
    dispatcher = _DISPATCHERS.get(intent_name, _dispatch_unsupported)
    return dispatcher(intent, analysis_payload, monthly_series, anomalies)
