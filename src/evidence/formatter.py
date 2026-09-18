"""Deterministic plain-text formatting of Evidence objects (V0.8.7),
for the Streamlit "Why this answer?" panel (V0.8.16). Mirrors qa_answer.py's
existing phrasing conventions — no LLM call, every value read verbatim.
"""


def _format_period_value(evidence: dict) -> str:
    source = evidence["source"]
    return f"{source.get('column')} in {source.get('period')} was {evidence.get('value')}."


def _format_period_extremum(evidence: dict) -> str:
    source = evidence["source"]
    label = {"min": "lowest", "max": "highest"}.get(evidence.get("metric"), "extreme")
    return f"The {label} value of {source.get('column')} was {evidence.get('value')}, in {source.get('period')}."


def _format_period_change(evidence: dict) -> str:
    source = evidence["source"]
    sentence = (
        f"{source.get('column')} changed from {evidence.get('previous')} in {source.get('from_period')} "
        f"to {evidence.get('current')} in {source.get('to_period')}"
    )
    if evidence.get("absolute_change") is not None:
        sentence += f", a change of {evidence['absolute_change']}"
        if evidence.get("percentage_change") is not None:
            sentence += f" ({evidence['percentage_change']}%)"
    return sentence + "."


def _format_column_stat(evidence: dict) -> str:
    source = evidence["source"]
    metric = evidence.get("metric") or "value"
    return f"The {metric} of {source.get('column')} is {evidence.get('value')}."


def _format_anomaly_summary(evidence: dict) -> str:
    source = evidence["source"]
    count = evidence.get("anomaly_count")
    column_suffix = f" for {source.get('column')}" if source.get("column") else ""
    if count == 0:
        return f"No anomalies were detected{column_suffix}."
    return f"{count} anomal{'y' if count == 1 else 'ies'} detected{column_suffix}."


def _format_missing_values(evidence: dict) -> str:
    columns = evidence.get("columns") or []
    if not columns:
        return "No columns have missing values."
    parts = [f"{c['name']} ({c['missing_count']} missing)" for c in columns]
    return "Columns with missing values: " + ", ".join(parts) + "."


def _format_trend(evidence: dict) -> str:
    source = evidence["source"]
    return f"{source.get('column')}'s trend is {evidence.get('trend')}."


def _format_file_comparison(evidence: dict) -> str:
    source = evidence["source"]
    sentence = (
        f"{source.get('column')} for {source.get('file_b') or 'the second file'} is {evidence.get('current')}, "
        f"compared with {evidence.get('previous')} for {source.get('file_a') or 'the first file'}"
    )
    if evidence.get("absolute_change") is not None:
        sentence += f", a change of {evidence['absolute_change']}"
    return sentence + "."


def _format_forecast(evidence: dict) -> str:
    source = evidence["source"]
    return (
        f"The baseline forecast for {source.get('column')} in {source.get('period')} "
        f"is {evidence.get('forecast')} (method: {evidence.get('method')}) — a prediction, not a historical fact."
    )


_FORMATTERS = {
    "period_value": _format_period_value,
    "period_extremum": _format_period_extremum,
    "period_change": _format_period_change,
    "column_stat": _format_column_stat,
    "anomaly_summary": _format_anomaly_summary,
    "missing_values": _format_missing_values,
    "trend": _format_trend,
    "file_comparison": _format_file_comparison,
    "forecast": _format_forecast,
}


def format_evidence(evidence: dict) -> str:
    formatter = _FORMATTERS.get(evidence.get("type"))
    if formatter is None:
        return "Supporting data is available but could not be summarized."
    return formatter(evidence)


def format_evidence_list(evidence_list: list[dict]) -> list[str]:
    return [format_evidence(evidence) for evidence in evidence_list]
