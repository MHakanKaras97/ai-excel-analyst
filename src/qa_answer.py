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
    # multi_file_comparison.dispatch_comparison_intent grounded-result failures (V0.7.5)
    "file_not_found": "I couldn't find a file matching that reference.",
    "insufficient_files": "At least two distinct files are needed for a comparison.",
    "role_not_assigned": "No file has been assigned that role yet.",
    "duplicate_role": "That reference matches more than one file — please be more specific.",
    "column_not_found_in_file_a": "I couldn't find that column in the first file.",
    "column_not_found_in_file_b": "I couldn't find that column in the second file.",
    "ambiguous_column_in_file_a": "That column reference is ambiguous in the first file.",
    "ambiguous_column_in_file_b": "That column reference is ambiguous in the second file.",
    "period_not_found_in_file_a": "I couldn't find that period in the first file.",
    "period_not_found_in_file_b": "I couldn't find that period in the second file.",
    "ambiguous_period_in_file_a": "That period reference is ambiguous in the first file.",
    "ambiguous_period_in_file_b": "That period reference is ambiguous in the second file.",
    "trend_not_computed_in_file_a": "No trend has been computed for the first file yet, so that comparison can't be answered.",
    "trend_not_computed_in_file_b": "No trend has been computed for the second file yet, so that comparison can't be answered.",
    "unsupported_comparison": "That comparison isn't supported yet.",
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


def _file_label(file_identity: dict | None) -> str:
    if not file_identity:
        return "the file"
    return file_identity.get("display_name") or file_identity.get("role") or "the file"


def _answer_file_value_comparison(result: dict) -> str:
    extra = result.get("extra") or {}
    metric = result.get("metric") or "value"
    column = result.get("column")
    sentence = (
        f"The {metric} of {column} for {_file_label(result.get('file_b'))} is {result['current']}, "
        f"compared with {result['previous']} for {_file_label(result.get('file_a'))}"
    )
    if result.get("absolute_change") is not None:
        sentence += f", a change of {result['absolute_change']}"
        if result.get("percentage_change") is not None:
            sentence += f" ({result['percentage_change']}%)"
    elif extra.get("compare_reason"):
        sentence += " (no change could be computed)"
    return sentence + "."


def _answer_file_period_comparison(result: dict) -> str:
    column = result.get("column")
    period = result.get("period")
    sentence = (
        f"{column} in {period} for {_file_label(result.get('file_b'))} is {result['current']}, "
        f"compared with {result['previous']} for {_file_label(result.get('file_a'))}"
    )
    if result.get("absolute_change") is not None:
        sentence += f", a change of {result['absolute_change']}"
        if result.get("percentage_change") is not None:
            sentence += f" ({result['percentage_change']}%)"
    return sentence + "."


def _answer_file_trend_comparison(result: dict) -> str:
    return (
        f"{_file_label(result.get('file_a'))}'s trend is {result['previous']}, while "
        f"{_file_label(result.get('file_b'))}'s trend is {result['current']}."
    )


def _answer_file_anomaly_comparison(result: dict) -> str:
    file_a_label = _file_label(result.get("file_a"))
    file_b_label = _file_label(result.get("file_b"))
    previous, current = result["previous"], result["current"]
    sentence = f"{file_a_label} has {previous} anomal{'y' if previous == 1 else 'ies'}, {file_b_label} has {current}"
    if result.get("absolute_change") is not None:
        sentence += f" — a change of {result['absolute_change']}"
    return sentence + "."


_COMPARISON_ANSWERERS = {
    "file_value_comparison": _answer_file_value_comparison,
    "file_period_comparison": _answer_file_period_comparison,
    "file_trend_comparison": _answer_file_trend_comparison,
    "file_anomaly_comparison": _answer_file_anomaly_comparison,
}


def answer_comparison_result(result: dict) -> str:
    """Produce a deterministic plain-text answer from a
    multi_file_comparison result (V0.7.5) — either a grounded Comparison
    Result (has a "found" key) or a multi-file interpreter failure (has an
    "is_valid" key that is False). Every value used is read as-is from the
    input; nothing is computed or invented here. Additive to this module:
    does not alter answer_grounded_result's existing single-file dispatch,
    since a Comparison Result uses "comparison_type" rather than "intent".
    """
    if not isinstance(result, dict):
        return DEFAULT_FAILURE_MESSAGE

    if "found" in result:
        if not result["found"]:
            return _failure_message(result)
        handler = _COMPARISON_ANSWERERS.get(result.get("comparison_type"))
        return handler(result) if handler else DEFAULT_FAILURE_MESSAGE

    if "is_valid" in result and not result["is_valid"]:
        return _failure_message(result)

    return DEFAULT_FAILURE_MESSAGE


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
