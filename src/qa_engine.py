import difflib
import re

import pandas as pd

# Conservative similarity cutoff for difflib.get_close_matches (0.0-1.0).
# The stdlib default (0.6) is loose enough to match weak references like
# "Rev" -> "Revenue"; 0.8 only accepts near-identical spellings (e.g. a
# single missing/typo'd character) and rejects short/partial references,
# so ambiguity is never silently guessed away. Adjust here if it proves
# too strict or too loose in practice.
FUZZY_CUTOFF = 0.8

MONTH_NAMES = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

PERIOD_LABEL_PATTERN = re.compile(r"^\d{4}-\d{2}$")


def _column_result(found: bool, reason, column=None, candidates=None) -> dict:
    return {
        "found": found,
        "reason": reason,
        "column": column,
        "candidates": candidates or [],
    }


def resolve_column(column_hint: str, column_names: list[str]) -> dict:
    """Resolve a free-text column hint against known column names.

    Resolution order: exact case-insensitive match, then (only if no exact
    match exists) a conservative fuzzy match via difflib. Ambiguity at
    either stage is reported, never silently guessed.
    """
    names = list(column_names)

    if not column_hint or not column_hint.strip():
        return _column_result(False, "column_not_found")

    hint = column_hint.strip()

    exact_matches = [name for name in names if name.lower() == hint.lower()]
    if len(exact_matches) == 1:
        return _column_result(True, None, column=exact_matches[0])
    if len(exact_matches) > 1:
        return _column_result(False, "ambiguous_column", candidates=exact_matches)

    if not names:
        return _column_result(False, "column_not_found")

    lower_names = [name.lower() for name in names]
    close = difflib.get_close_matches(hint.lower(), lower_names, n=len(lower_names), cutoff=FUZZY_CUTOFF)

    candidates = []
    for lower_match in close:
        for name in names:
            if name.lower() == lower_match and name not in candidates:
                candidates.append(name)

    if len(candidates) == 1:
        return _column_result(True, None, column=candidates[0])
    if len(candidates) > 1:
        return _column_result(False, "ambiguous_column", candidates=candidates)

    return _column_result(False, "column_not_found")


def _period_result(found: bool, reason, period=None, candidates=None) -> dict:
    return {
        "found": found,
        "reason": reason,
        "period": period,
        "candidates": candidates or [],
    }


def _month_number(token: str):
    token_lower = token.lower()
    if token_lower in MONTH_NAMES:
        return MONTH_NAMES[token_lower]
    if token.isdigit() and 1 <= int(token) <= 12:
        return int(token)
    return None


def _parse_month_and_year(hint: str):
    """Parse a two-token "<month> <year>" hint (e.g. "March 2024", "Mar 2024",
    "03 2024") into (month, year) ints. Returns None for any other shape."""
    parts = hint.split()
    if len(parts) != 2:
        return None

    month_token, year_token = parts
    if not (year_token.isdigit() and len(year_token) == 4):
        return None

    month_num = _month_number(month_token)
    if month_num is None:
        return None

    return month_num, int(year_token)


def _month_of_label(label: str) -> int:
    return int(label[5:7])


def resolve_period(period_hint: str, period_labels: list[str]) -> dict:
    """Resolve a free-text period hint against known "YYYY-MM" period labels.

    Resolution order: exact known label, then a fully-specified month+year
    normalized to "YYYY-MM", then a bare month name/number matched across
    all known years. Ambiguity is always reported, never defaulted to the
    latest/first candidate.
    """
    labels = list(period_labels)

    if not period_hint or not period_hint.strip():
        return _period_result(False, "period_not_found")

    hint = period_hint.strip()

    if hint in labels:
        return _period_result(True, None, period=hint)

    month_year = _parse_month_and_year(hint)
    if month_year is not None:
        month_num, year = month_year
        candidate = f"{year:04d}-{month_num:02d}"
        if candidate in labels:
            return _period_result(True, None, period=candidate)
        return _period_result(False, "period_not_found")

    if len(hint.split()) == 1:
        month_num = _month_number(hint)
        if month_num is not None:
            matches = sorted(label for label in labels if PERIOD_LABEL_PATTERN.match(label) and _month_of_label(label) == month_num)
            if len(matches) == 1:
                return _period_result(True, None, period=matches[0])
            if len(matches) > 1:
                return _period_result(False, "ambiguous_period", candidates=matches)
            return _period_result(False, "period_not_found")

    return _period_result(False, "period_not_found")


# ==================================================
# INTENT DISPATCH (V0.5.3b)
# ==================================================

COLUMN_STAT_METRICS = {"sum", "mean", "median", "min", "max", "count"}
PERIOD_EXTREMUM_METRICS = {"min", "max"}


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


def _grounded_result(found: bool, reason, intent_name: str, column=None, period=None, value=None, extra=None) -> dict:
    return {
        "found": found,
        "reason": reason,
        "intent": intent_name,
        "column": column,
        "period": period,
        "value": _json_safe(value),
        "extra": extra or {},
    }


def _numeric_summary_columns(analysis_payload: dict) -> list:
    return (analysis_payload.get("numeric_summary") or {}).get("columns") or []


def _profile_columns(analysis_payload: dict) -> list:
    return (analysis_payload.get("profile") or {}).get("columns") or []


def _resolve_trend_column(column_hint, analysis_payload: dict, monthly_series):
    """Resolve an optional column hint against the known numeric columns.

    Period-based intents operate on the single, already-selected trend
    series, so a hint is not required — when absent, the series' own name
    (if the caller set one) is used purely for reporting, not resolution.

    When a hint IS given and resolves to a real column, that column must
    still refer to the same series the caller is about to query — otherwise
    the result would report one column's name while reading another
    column's value (e.g. hint "Revenue" resolving successfully, but the
    value actually read comes from a "TotalPrice" trend series). If
    `monthly_series` carries a name, a resolved column that disagrees with
    it is rejected as "column_series_mismatch" rather than silently
    answered against the wrong series. A nameless series (as currently
    produced by app.py's build_monthly_series) can't be checked and is
    passed through unchanged.

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


def _dispatch_period_value(intent: dict, analysis_payload: dict, monthly_series) -> dict:
    if monthly_series is None or len(monthly_series) == 0:
        return _grounded_result(False, "trend_not_computed", "period_value")

    ok, column, reason, candidates = _resolve_trend_column(intent.get("column_hint"), analysis_payload, monthly_series)
    if not ok:
        return _grounded_result(False, reason, "period_value", extra={"candidates": candidates} if candidates else None)

    period_result = resolve_period(intent.get("period_hint"), list(monthly_series.index))
    if not period_result["found"]:
        candidates = period_result["candidates"]
        return _grounded_result(False, period_result["reason"], "period_value", extra={"candidates": candidates} if candidates else None)

    period = period_result["period"]
    value = monthly_series.loc[period]
    return _grounded_result(True, None, "period_value", column=column, period=period, value=value)


def _dispatch_period_extremum(intent: dict, analysis_payload: dict, monthly_series) -> dict:
    metric = intent.get("metric")
    if metric not in PERIOD_EXTREMUM_METRICS:
        return _grounded_result(False, "unsupported", "period_extremum")

    if monthly_series is None or len(monthly_series) == 0:
        return _grounded_result(False, "trend_not_computed", "period_extremum")

    ok, column, reason, candidates = _resolve_trend_column(intent.get("column_hint"), analysis_payload, monthly_series)
    if not ok:
        return _grounded_result(False, reason, "period_extremum", extra={"candidates": candidates} if candidates else None)

    period = monthly_series.idxmin() if metric == "min" else monthly_series.idxmax()
    value = monthly_series.loc[period]
    return _grounded_result(True, None, "period_extremum", column=column, period=period, value=value, extra={"metric": metric})


def _dispatch_period_change(intent: dict, analysis_payload: dict, monthly_series) -> dict:
    period_comparison = analysis_payload.get("period_comparison")
    if not period_comparison:
        return _grounded_result(False, "trend_not_computed", "period_change")

    ok, column, reason, candidates = _resolve_trend_column(intent.get("column_hint"), analysis_payload, monthly_series)
    if not ok:
        return _grounded_result(False, reason, "period_change", extra={"candidates": candidates} if candidates else None)

    known_periods = list(period_comparison.get("periods") or [])

    from_result = resolve_period(intent.get("from_period_hint"), known_periods)
    if not from_result["found"]:
        candidates = from_result["candidates"]
        return _grounded_result(False, from_result["reason"], "period_change", extra={"candidates": candidates} if candidates else None)

    to_result = resolve_period(intent.get("to_period_hint"), known_periods)
    if not to_result["found"]:
        candidates = to_result["candidates"]
        return _grounded_result(False, to_result["reason"], "period_change", extra={"candidates": candidates} if candidates else None)

    from_period = from_result["period"]
    to_period = to_result["period"]

    comparison = next(
        (
            c for c in period_comparison.get("comparisons") or []
            if c.get("from_period") == from_period and c.get("to_period") == to_period
        ),
        None,
    )
    if comparison is None:
        return _grounded_result(
            False, "period_not_found", "period_change",
            extra={"from_period": from_period, "to_period": to_period},
        )

    extra = {
        "from_period": comparison.get("from_period"),
        "to_period": comparison.get("to_period"),
        "previous": comparison.get("previous"),
        "current": comparison.get("current"),
        "absolute_change": comparison.get("absolute_change"),
        "percentage_change": comparison.get("percentage_change"),
        "is_valid": comparison.get("is_valid"),
    }
    return _grounded_result(True, None, "period_change", column=column, value=comparison.get("absolute_change"), extra=extra)


def _dispatch_column_stat(intent: dict, analysis_payload: dict, monthly_series) -> dict:
    metric = intent.get("metric")
    if metric not in COLUMN_STAT_METRICS:
        return _grounded_result(False, "unsupported", "column_stat")

    column_names = [c["name"] for c in _numeric_summary_columns(analysis_payload)]
    result = resolve_column(intent.get("column_hint"), column_names)
    if not result["found"]:
        candidates = result["candidates"]
        return _grounded_result(False, result["reason"], "column_stat", extra={"candidates": candidates} if candidates else None)

    column = result["column"]
    column_stats = next((c for c in _numeric_summary_columns(analysis_payload) if c["name"] == column), None)
    if column_stats is None:
        return _grounded_result(False, "column_not_found", "column_stat")

    return _grounded_result(True, None, "column_stat", column=column, value=column_stats.get(metric), extra={"metric": metric})


def _dispatch_anomaly_check(intent: dict, analysis_payload: dict, monthly_series) -> dict:
    anomalies = analysis_payload.get("anomalies")
    if not anomalies:
        return _grounded_result(False, "trend_not_computed", "anomaly_check")

    ok, column, reason, candidates = _resolve_trend_column(intent.get("column_hint"), analysis_payload, monthly_series)
    if not ok:
        return _grounded_result(False, reason, "anomaly_check", extra={"candidates": candidates} if candidates else None)

    extra = {
        "insufficient_data": anomalies.get("insufficient_data"),
        "lower_bound": anomalies.get("lower_bound"),
        "upper_bound": anomalies.get("upper_bound"),
        "anomalies": list(anomalies.get("anomalies") or []),
    }
    return _grounded_result(True, None, "anomaly_check", column=column, value=anomalies.get("anomaly_count"), extra=extra)


def _dispatch_missing_values(intent: dict, analysis_payload: dict, monthly_series) -> dict:
    profile = analysis_payload.get("profile")
    if not profile:
        return _grounded_result(False, "profile_not_computed", "missing_values")

    columns_with_missing = [
        {"name": c["name"], "missing_count": c["missing_count"], "missing_percentage": c["missing_percentage"]}
        for c in _profile_columns(analysis_payload)
        if c.get("missing_count", 0) > 0
    ]
    return _grounded_result(
        True, None, "missing_values",
        value=len(columns_with_missing),
        extra={"columns": columns_with_missing},
    )


def _dispatch_unsupported(intent: dict, analysis_payload: dict, monthly_series) -> dict:
    return _grounded_result(False, "unsupported", "unsupported")


_DISPATCHERS = {
    "period_value": _dispatch_period_value,
    "period_extremum": _dispatch_period_extremum,
    "period_change": _dispatch_period_change,
    "column_stat": _dispatch_column_stat,
    "anomaly_check": _dispatch_anomaly_check,
    "missing_values": _dispatch_missing_values,
    "unsupported": _dispatch_unsupported,
}


def dispatch_intent(intent: dict, analysis_payload: dict, monthly_series=None) -> dict:
    """Dispatch a validated structured intent against already-computed analysis
    data and return a grounded result. Purely deterministic: no LLM call, no
    new analytics, no mutation of any input.

    `analysis_payload` is expected to hold whichever of the already-computed
    keys are relevant to the intent — "numeric_summary", "period_comparison",
    "anomalies", "profile" — exactly as produced by the existing analytics/
    anomaly/profiling modules. `monthly_series` is the already-computed
    trend Series (as built by app.py's build_monthly_series) for period-based
    intents.
    """
    intent_name = intent.get("intent")
    dispatcher = _DISPATCHERS.get(intent_name, _dispatch_unsupported)
    return dispatcher(intent, analysis_payload, monthly_series)
