"""Deterministic multi-file comparison engine (V0.7.3).

No Streamlit, no LLM calls. Resolves an explicit comparison request (which
two files, which column/period/metric) against already-computed
FileAnalysis data and produces a fully-keyed, JSON-safe Comparison Result.

Column and column/series-mismatch resolution are delegated entirely to
qa_engine.resolve_column()/resolve_trend_column(); period resolution to
qa_engine.resolve_period() — none of these are reimplemented here. The
actual previous-vs-current arithmetic is delegated entirely to
analytics_engine.compare_values() — no second percentage-change
calculation exists in this module. Trend/anomaly comparisons read only
already-computed FileAnalysis fields — nothing is recalculated.

Strictly pairwise: every comparison concerns exactly two files, resolved
independently by explicit hint (role or display_name, exact match only —
never fuzzy, never inferred from upload order). There is no path to select
more than two files, so a 3+-file request is structurally unsupported
rather than something this module has to specifically detect and reject.
"""
from src.analytics_engine import compare_values
from src.qa_engine import resolve_column, resolve_period, resolve_trend_column

RECOGNIZED_ROLES = {"previous", "current", "reference", "comparison"}

SUPPORTED_COMPARISON_TYPES = {
    "file_value_comparison",
    "file_period_comparison",
    "file_trend_comparison",
    "file_anomaly_comparison",
}


# ==================================================
# FILE RESOLUTION
# ==================================================


def _file_result(found: bool, reason, file_id=None, candidates=None) -> dict:
    return {
        "found": found,
        "reason": reason,
        "file_id": file_id,
        "candidates": candidates or [],
    }


def resolve_file(hint, data_files: dict) -> dict:
    """Resolve a free-text file hint against the registered DataFile records.

    Exact match only: tried against explicitly assigned roles first (from
    RECOGNIZED_ROLES), then against display_name. Never fuzzy, and file
    identity/roles are never inferred — only what was explicitly assigned
    to a DataFile is considered. Picking the wrong file silently is a worse
    failure than a column typo, so this is deliberately stricter than
    resolve_column() (no fuzzy fallback at all).

    A hint matching more than one file — whether because two files share
    the same assigned role or the same display_name — is reported as
    "duplicate_role"; there is no separate "ambiguous file" reason in the
    approved vocabulary, and both situations mean the same thing here: the
    hint does not pick out a single file.
    """
    if not hint or not hint.strip():
        return _file_result(False, "file_not_found")

    hint_normalized = hint.strip().lower()

    if hint_normalized in RECOGNIZED_ROLES:
        matches = [
            file_id for file_id, data_file in data_files.items()
            if (data_file.get("role") or "").lower() == hint_normalized
        ]
        if len(matches) == 1:
            return _file_result(True, None, file_id=matches[0])
        if len(matches) > 1:
            return _file_result(False, "duplicate_role", candidates=matches)
        return _file_result(False, "role_not_assigned")

    matches = [
        file_id for file_id, data_file in data_files.items()
        if (data_file.get("display_name") or "").lower() == hint_normalized
    ]
    if len(matches) == 1:
        return _file_result(True, None, file_id=matches[0])
    if len(matches) > 1:
        return _file_result(False, "duplicate_role", candidates=matches)
    return _file_result(False, "file_not_found")


def _identity(data_file: dict) -> dict:
    return {
        "file_id": data_file["file_id"],
        "display_name": data_file["display_name"],
        "role": data_file["role"],
    }


# ==================================================
# COMPARISON RESULT CONTRACT
# ==================================================


def _result(found: bool, reason, comparison_type, metric=None, column=None, period=None,
            file_a=None, file_b=None, previous=None, current=None, absolute_change=None,
            percentage_change=None, is_valid=None, extra=None) -> dict:
    return {
        "found": found,
        "reason": reason,
        "comparison_type": comparison_type,
        "metric": metric,
        "column": column,
        "period": period,
        "file_a": file_a,
        "file_b": file_b,
        "previous": previous,
        "current": current,
        "absolute_change": absolute_change,
        "percentage_change": percentage_change,
        "is_valid": is_valid,
        "extra": extra or {},
    }


def _numeric_summary_columns(analysis) -> list:
    return ((analysis or {}).get("numeric_summary") or {}).get("columns") or []


def _resolve_value_column(column_hint, analysis) -> dict:
    column_names = [c["name"] for c in _numeric_summary_columns(analysis)]
    return resolve_column(column_hint, column_names)


def _suffix_column_reason(reason: str, which: str) -> str:
    """Map a generic resolve_column()/resolve_trend_column() reason to its
    per-file variant. "column_series_mismatch" is not file-specific in the
    approved reason vocabulary and is passed through unchanged."""
    if reason in ("column_not_found", "ambiguous_column"):
        return f"{reason}_in_file_{which}"
    return reason


def _suffix_period_reason(reason: str, which: str) -> str:
    return f"{reason}_in_file_{which}"


def _compare_value(comparison_type, file_a_analysis, file_b_analysis, column_hint, metric) -> dict:
    result_a = _resolve_value_column(column_hint, file_a_analysis)
    if not result_a["found"]:
        return _result(
            False, _suffix_column_reason(result_a["reason"], "a"), comparison_type, metric=metric,
            extra={"candidates": result_a["candidates"]} if result_a["candidates"] else None,
        )

    result_b = _resolve_value_column(column_hint, file_b_analysis)
    if not result_b["found"]:
        return _result(
            False, _suffix_column_reason(result_b["reason"], "b"), comparison_type, metric=metric,
            extra={"candidates": result_b["candidates"]} if result_b["candidates"] else None,
        )

    column_a, column_b = result_a["column"], result_b["column"]
    entry_a = next((c for c in _numeric_summary_columns(file_a_analysis) if c["name"] == column_a), None)
    entry_b = next((c for c in _numeric_summary_columns(file_b_analysis) if c["name"] == column_b), None)
    if entry_a is None:
        return _result(False, "column_not_found_in_file_a", comparison_type, metric=metric)
    if entry_b is None:
        return _result(False, "column_not_found_in_file_b", comparison_type, metric=metric)

    comparison = compare_values(entry_a.get(metric), entry_b.get(metric))

    extra = {}
    if comparison["reason"]:
        extra["compare_reason"] = comparison["reason"]
    if column_b != column_a:
        extra["column_file_b"] = column_b

    return _result(
        True, None, comparison_type, metric=metric, column=column_a,
        previous=comparison["previous"], current=comparison["current"],
        absolute_change=comparison["absolute_change"], percentage_change=comparison["percentage_change"],
        is_valid=comparison["is_valid"], extra=extra or None,
    )


def _compare_period(comparison_type, file_a_analysis, file_b_analysis, column_hint, period_hint) -> dict:
    series_a = (file_a_analysis or {}).get("monthly_series")
    series_b = (file_b_analysis or {}).get("monthly_series")

    if series_a is None or len(series_a) == 0:
        return _result(False, "trend_not_computed_in_file_a", comparison_type)
    if series_b is None or len(series_b) == 0:
        return _result(False, "trend_not_computed_in_file_b", comparison_type)

    ok_a, column_a, reason_a, candidates_a = resolve_trend_column(column_hint, file_a_analysis, series_a)
    if not ok_a:
        return _result(
            False, _suffix_column_reason(reason_a, "a"), comparison_type,
            extra={"candidates": candidates_a} if candidates_a else None,
        )

    ok_b, column_b, reason_b, candidates_b = resolve_trend_column(column_hint, file_b_analysis, series_b)
    if not ok_b:
        return _result(
            False, _suffix_column_reason(reason_b, "b"), comparison_type,
            extra={"candidates": candidates_b} if candidates_b else None,
        )

    period_result_a = resolve_period(period_hint, list(series_a.index))
    if not period_result_a["found"]:
        return _result(
            False, _suffix_period_reason(period_result_a["reason"], "a"), comparison_type,
            extra={"candidates": period_result_a["candidates"]} if period_result_a["candidates"] else None,
        )

    period_result_b = resolve_period(period_hint, list(series_b.index))
    if not period_result_b["found"]:
        return _result(
            False, _suffix_period_reason(period_result_b["reason"], "b"), comparison_type,
            extra={"candidates": period_result_b["candidates"]} if period_result_b["candidates"] else None,
        )

    period_a, period_b = period_result_a["period"], period_result_b["period"]
    comparison = compare_values(series_a.loc[period_a], series_b.loc[period_b])

    extra = {}
    if comparison["reason"]:
        extra["compare_reason"] = comparison["reason"]
    if period_b != period_a:
        extra["period_file_b"] = period_b

    return _result(
        True, None, comparison_type, column=column_a, period=period_a,
        previous=comparison["previous"], current=comparison["current"],
        absolute_change=comparison["absolute_change"], percentage_change=comparison["percentage_change"],
        is_valid=comparison["is_valid"], extra=extra or None,
    )


def _compare_trend(comparison_type, file_a_analysis, file_b_analysis, column_hint) -> dict:
    trend_a = (file_a_analysis or {}).get("trend")
    trend_b = (file_b_analysis or {}).get("trend")
    if not trend_a:
        return _result(False, "trend_not_computed_in_file_a", comparison_type)
    if not trend_b:
        return _result(False, "trend_not_computed_in_file_b", comparison_type)

    series_a = (file_a_analysis or {}).get("monthly_series")
    series_b = (file_b_analysis or {}).get("monthly_series")

    ok_a, column_a, reason_a, candidates_a = resolve_trend_column(column_hint, file_a_analysis, series_a)
    if not ok_a:
        return _result(
            False, _suffix_column_reason(reason_a, "a"), comparison_type,
            extra={"candidates": candidates_a} if candidates_a else None,
        )

    ok_b, column_b, reason_b, candidates_b = resolve_trend_column(column_hint, file_b_analysis, series_b)
    if not ok_b:
        return _result(
            False, _suffix_column_reason(reason_b, "b"), comparison_type,
            extra={"candidates": candidates_b} if candidates_b else None,
        )

    return _result(
        True, None, comparison_type, column=column_a,
        previous=trend_a.get("trend"), current=trend_b.get("trend"),
        absolute_change=None, percentage_change=None, is_valid=True,
        extra={"file_a_trend": trend_a, "file_b_trend": trend_b},
    )


def _compare_anomaly(comparison_type, file_a_analysis, file_b_analysis, column_hint) -> dict:
    anomalies_a = (file_a_analysis or {}).get("anomalies")
    anomalies_b = (file_b_analysis or {}).get("anomalies")
    if not anomalies_a:
        return _result(False, "trend_not_computed_in_file_a", comparison_type)
    if not anomalies_b:
        return _result(False, "trend_not_computed_in_file_b", comparison_type)

    series_a = (file_a_analysis or {}).get("monthly_series")
    series_b = (file_b_analysis or {}).get("monthly_series")

    ok_a, column_a, reason_a, candidates_a = resolve_trend_column(column_hint, file_a_analysis, series_a)
    if not ok_a:
        return _result(
            False, _suffix_column_reason(reason_a, "a"), comparison_type,
            extra={"candidates": candidates_a} if candidates_a else None,
        )

    ok_b, column_b, reason_b, candidates_b = resolve_trend_column(column_hint, file_b_analysis, series_b)
    if not ok_b:
        return _result(
            False, _suffix_column_reason(reason_b, "b"), comparison_type,
            extra={"candidates": candidates_b} if candidates_b else None,
        )

    comparison = compare_values(anomalies_a.get("anomaly_count"), anomalies_b.get("anomaly_count"))

    extra = {"file_a_anomalies": anomalies_a, "file_b_anomalies": anomalies_b}
    if comparison["reason"]:
        extra["compare_reason"] = comparison["reason"]

    return _result(
        True, None, comparison_type, column=column_a,
        previous=comparison["previous"], current=comparison["current"],
        absolute_change=comparison["absolute_change"], percentage_change=comparison["percentage_change"],
        is_valid=comparison["is_valid"], extra=extra,
    )


_COMPARATORS = {
    "file_value_comparison": lambda t, a, b, column_hint, metric, period_hint: _compare_value(t, a, b, column_hint, metric),
    "file_period_comparison": lambda t, a, b, column_hint, metric, period_hint: _compare_period(t, a, b, column_hint, period_hint),
    "file_trend_comparison": lambda t, a, b, column_hint, metric, period_hint: _compare_trend(t, a, b, column_hint),
    "file_anomaly_comparison": lambda t, a, b, column_hint, metric, period_hint: _compare_anomaly(t, a, b, column_hint),
}


def compare_files(comparison_type: str, file_a_analysis: dict, file_b_analysis: dict,
                   column_hint=None, metric=None, period_hint=None) -> dict:
    """Compare two already-computed FileAnalysis dicts deterministically.

    Column/period resolution is delegated to qa_engine.resolve_column()/
    resolve_period()/resolve_trend_column(); the previous-vs-current
    arithmetic is delegated to analytics_engine.compare_values() — neither
    is reimplemented here. This function is not file-identity-aware (it
    only ever sees FileAnalysis dicts, never file_id/display_name/role) —
    dispatch_comparison_intent() adds "file_a"/"file_b" identity afterward.
    """
    comparator = _COMPARATORS.get(comparison_type)
    if comparator is None:
        return _result(False, "unsupported_comparison", comparison_type)
    return comparator(comparison_type, file_a_analysis, file_b_analysis, column_hint, metric, period_hint)


def dispatch_comparison_intent(intent: dict, data_files: dict) -> dict:
    """Resolve a structured multi-file comparison intent (matching the
    approved V0.7 schema: intent, column_hint, metric, period_hint,
    from_file_hint, to_file_hint) against the currently registered DataFile
    records and return a fully-keyed, JSON-safe Comparison Result.

    Strictly pairwise: from_file_hint/to_file_hint are resolved
    independently via resolve_file(); there is no mechanism to select more
    than two files, so a 3+-file comparison is never possible to request
    through this schema in the first place.
    """
    comparison_type = intent.get("intent")

    if comparison_type not in SUPPORTED_COMPARISON_TYPES:
        return _result(False, "unsupported_comparison", comparison_type)

    if len(data_files) < 2:
        return _result(False, "insufficient_files", comparison_type)

    from_result = resolve_file(intent.get("from_file_hint"), data_files)
    if not from_result["found"]:
        return _result(
            False, from_result["reason"], comparison_type,
            extra={"candidates": from_result["candidates"]} if from_result["candidates"] else None,
        )

    to_result = resolve_file(intent.get("to_file_hint"), data_files)
    if not to_result["found"]:
        return _result(
            False, to_result["reason"], comparison_type,
            extra={"candidates": to_result["candidates"]} if to_result["candidates"] else None,
        )

    file_a_id, file_b_id = from_result["file_id"], to_result["file_id"]
    if file_a_id == file_b_id:
        # The two hints resolved to the very same file — a comparison needs
        # two distinct files, and there is no dedicated "same file" reason
        # in the approved vocabulary; this is treated the same as not
        # having two usable files to compare.
        return _result(False, "insufficient_files", comparison_type)

    data_file_a, data_file_b = data_files[file_a_id], data_files[file_b_id]

    comparison = compare_files(
        comparison_type,
        data_file_a.get("analysis"),
        data_file_b.get("analysis"),
        column_hint=intent.get("column_hint"),
        metric=intent.get("metric"),
        period_hint=intent.get("period_hint"),
    )
    comparison["file_a"] = _identity(data_file_a)
    comparison["file_b"] = _identity(data_file_b)
    return comparison
