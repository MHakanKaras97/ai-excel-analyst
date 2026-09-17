"""Deterministic phrasing layer for the Q&A pipeline.

Turns a qa_engine.dispatch_intent() grounded result (or a qa_interpreter
failure, for when interpretation itself never reached the engine) into a
plain-text answer. No LLM call, no calculation, no new data lookup — every
number in the output comes verbatim from the input dict. Phrasing lives here
in isolation so it can be revised without touching the interpreter or engine.
"""

FAILURE_MESSAGES = {
    # qa_engine.dispatch_intent grounded-result failures
    "column_not_found": "I couldn't find a column matching that question.",
    "ambiguous_column": "That column reference is ambiguous.",
    "period_not_found": "I couldn't find a period matching that question.",
    "ambiguous_period": "That period reference is ambiguous.",
    "trend_not_computed": "No trend has been computed for this dataset yet, so that question can't be answered.",
    "profile_not_computed": "No data profile has been computed for this dataset yet, so that question can't be answered.",
    "column_series_mismatch": "That column doesn't match the currently selected trend series.",
    "unsupported": "That question isn't supported yet.",
    # qa_interpreter.interpret_question failures (question never reached the engine)
    "invalid_json": "The question couldn't be interpreted (the AI response wasn't valid JSON).",
    "invalid_schema": "The question couldn't be interpreted (the AI response didn't match the expected format).",
    "provider_error": "The question couldn't be interpreted because the AI provider returned an error.",
}

DEFAULT_FAILURE_MESSAGE = "That question couldn't be answered."


def _candidates_suffix(result: dict) -> str:
    candidates = (result.get("extra") or {}).get("candidates")
    if not candidates:
        return ""
    return f" Candidates: {', '.join(str(c) for c in candidates)}."


def _failure_message(result: dict) -> str:
    message = FAILURE_MESSAGES.get(result.get("reason"), DEFAULT_FAILURE_MESSAGE)
    return message + _candidates_suffix(result)


def _answer_period_value(result: dict) -> str:
    return f"{result['column']} in {result['period']} was {result['value']}."


def _answer_period_extremum(result: dict) -> str:
    metric = (result.get("extra") or {}).get("metric")
    label = {"min": "lowest", "max": "highest"}.get(metric, "extreme")
    return f"The {label} value of {result['column']} was {result['value']}, in {result['period']}."


def _answer_period_change(result: dict) -> str:
    extra = result.get("extra") or {}
    sentence = (
        f"{result['column']} changed from {extra.get('previous')} in {extra.get('from_period')} "
        f"to {extra.get('current')} in {extra.get('to_period')}, a change of {extra.get('absolute_change')}"
    )
    percentage_change = extra.get("percentage_change")
    if percentage_change is not None:
        sentence += f" ({percentage_change}%)"
    return sentence + "."


def _answer_column_stat(result: dict) -> str:
    metric = (result.get("extra") or {}).get("metric") or "value"
    return f"The {metric} of {result['column']} is {result['value']}."


def _answer_anomaly_check(result: dict) -> str:
    extra = result.get("extra") or {}
    column_suffix = f" for {result['column']}" if result.get("column") else ""

    if extra.get("insufficient_data"):
        return f"There is not enough data{column_suffix} to reliably check for anomalies."

    count = result.get("value")
    if count == 0:
        return f"No anomalies were detected{column_suffix}."
    return f"{count} anomal{'y' if count == 1 else 'ies'} detected{column_suffix}."


def _answer_missing_values(result: dict) -> str:
    columns = (result.get("extra") or {}).get("columns") or []
    if not columns:
        return "No columns have missing values."
    parts = [f"{c['name']} ({c['missing_count']} missing)" for c in columns]
    return "Columns with missing values: " + ", ".join(parts) + "."


_SUCCESS_ANSWERERS = {
    "period_value": _answer_period_value,
    "period_extremum": _answer_period_extremum,
    "period_change": _answer_period_change,
    "column_stat": _answer_column_stat,
    "anomaly_check": _answer_anomaly_check,
    "missing_values": _answer_missing_values,
}


def answer_grounded_result(result: dict) -> str:
    """Produce a deterministic plain-text answer from a pipeline result.

    Accepts either a qa_engine.dispatch_intent() grounded result (has a
    "found" key) or a qa_interpreter.interpret_question() failure (has an
    "is_valid" key that is False) — the two points where the Q&A pipeline
    can terminate before a human-readable answer exists. Every value used
    is read as-is from the input; nothing is computed or invented here.
    """
    if not isinstance(result, dict):
        return DEFAULT_FAILURE_MESSAGE

    if "found" in result:
        if not result["found"]:
            return _failure_message(result)
        handler = _SUCCESS_ANSWERERS.get(result.get("intent"))
        return handler(result) if handler else DEFAULT_FAILURE_MESSAGE

    if "is_valid" in result and not result["is_valid"]:
        return _failure_message(result)

    return DEFAULT_FAILURE_MESSAGE
