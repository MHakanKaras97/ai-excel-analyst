"""Deterministically build Evidence objects from already-computed results
(V0.8.7).

Two entry points:

- `build_evidence_from_grounded_result()` — for a single-file
  qa_engine.dispatch_intent() result.
- `build_evidence_from_comparison_result()` — for a multi_file_comparison.
  dispatch_comparison_intent() result.

Both are pure re-shaping: every value placed into an Evidence object is
read verbatim from the already-computed result, never recalculated. A
failed (`found: False`) result produces no evidence at all — there is
nothing deterministic to attest to.
"""
from src.evidence.models import make_evidence, make_evidence_id


def _source(file_label: str | None, column=None, period=None, **extra) -> dict:
    source = {}
    if file_label is not None:
        source["file"] = file_label
    if column is not None:
        source["column"] = column
    if period is not None:
        source["period"] = period
    source.update({k: v for k, v in extra.items() if v is not None})
    return source


def _build_period_value(result: dict, file_label, next_id) -> list:
    return [make_evidence(
        next_id(), "period_value",
        _source(file_label, column=result.get("column"), period=result.get("period")),
        value=result.get("value"),
    )]


def _build_period_extremum(result: dict, file_label, next_id) -> list:
    extra = result.get("extra") or {}
    return [make_evidence(
        next_id(), "period_extremum",
        _source(file_label, column=result.get("column"), period=result.get("period")),
        value=result.get("value"), metric=extra.get("metric"),
    )]


def _build_period_change(result: dict, file_label, next_id) -> list:
    extra = result.get("extra") or {}
    # qa_engine.dispatch_intent()'s period_change extra dict carries
    # "is_valid" (from analytics_engine.compare_values()) but never copies
    # that same call's "reason" (e.g. "division_by_zero") into it — that is
    # existing V0.7 grounded-result shape, left as-is here. Only the field
    # that actually reaches this function (is_valid) is propagated.
    return [make_evidence(
        next_id(), "period_change",
        _source(
            file_label, column=result.get("column"),
            from_period=extra.get("from_period"), to_period=extra.get("to_period"),
        ),
        previous=extra.get("previous"), current=extra.get("current"),
        absolute_change=extra.get("absolute_change"), percentage_change=extra.get("percentage_change"),
        is_valid=extra.get("is_valid"),
    )]


def _build_column_stat(result: dict, file_label, next_id) -> list:
    extra = result.get("extra") or {}
    return [make_evidence(
        next_id(), "column_stat",
        _source(file_label, column=result.get("column")),
        value=result.get("value"), metric=extra.get("metric"),
    )]


def _build_anomaly_check(result: dict, file_label, next_id) -> list:
    extra = result.get("extra") or {}
    return [make_evidence(
        next_id(), "anomaly_summary",
        _source(file_label, column=result.get("column")),
        anomaly_count=result.get("value"),
        lower_bound=extra.get("lower_bound"), upper_bound=extra.get("upper_bound"),
        anomalies=list(extra.get("anomalies") or []),
    )]


def _build_missing_values(result: dict, file_label, next_id) -> list:
    extra = result.get("extra") or {}
    return [make_evidence(
        next_id(), "missing_values",
        _source(file_label),
        column_count_with_missing=result.get("value"),
        columns=list(extra.get("columns") or []),
    )]


_GROUNDED_BUILDERS = {
    "period_value": _build_period_value,
    "period_extremum": _build_period_extremum,
    "period_change": _build_period_change,
    "column_stat": _build_column_stat,
    "anomaly_check": _build_anomaly_check,
    "missing_values": _build_missing_values,
}


def build_evidence_from_grounded_result(result: dict, file_label: str | None = None,
                                         start_index: int = 1) -> list[dict]:
    """Build zero or more Evidence dicts from a qa_engine.dispatch_intent()
    grounded result. Returns [] for a failed result or an intent with no
    registered evidence builder (e.g. "unsupported")."""
    if not isinstance(result, dict) or not result.get("found"):
        return []

    builder = _GROUNDED_BUILDERS.get(result.get("intent"))
    if builder is None:
        return []

    counter = {"next": start_index}

    def next_id():
        evidence_id = make_evidence_id(counter["next"])
        counter["next"] += 1
        return evidence_id

    return builder(result, file_label, next_id)


def build_evidence_from_comparison_result(result: dict, start_index: int = 1) -> list[dict]:
    """Build one "file_comparison" Evidence dict from a
    multi_file_comparison.dispatch_comparison_intent() result. Returns []
    for a failed (`found: False`) result."""
    if not isinstance(result, dict) or not result.get("found"):
        return []

    def _identity_label(identity):
        if not identity:
            return None
        return identity.get("display_name") or identity.get("role")

    source = _source(
        None, column=result.get("column"), period=result.get("period"),
        file_a=_identity_label(result.get("file_a")), file_b=_identity_label(result.get("file_b")),
    )

    # multi_file_comparison's Comparison Result keeps the underlying
    # compare_values() "reason" (e.g. "division_by_zero") in
    # extra["compare_reason"] rather than the top-level "reason" field
    # (which is reserved for comparison-level failures like
    # "file_not_found" and is always None on a found=True result) — both
    # are propagated here under their own names.
    extra = result.get("extra") or {}
    return [make_evidence(
        make_evidence_id(start_index), "file_comparison", source,
        comparison_type=result.get("comparison_type"), metric=result.get("metric"),
        previous=result.get("previous"), current=result.get("current"),
        absolute_change=result.get("absolute_change"), percentage_change=result.get("percentage_change"),
        is_valid=result.get("is_valid"), compare_reason=extra.get("compare_reason"),
    )]
