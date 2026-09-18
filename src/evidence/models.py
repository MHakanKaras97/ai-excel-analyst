"""Evidence / provenance model (V0.8.7).

An Evidence object is a small, fully-keyed, JSON-safe record of one
deterministic fact — never something an LLM is allowed to author. Evidence
is always built (src/evidence/builder.py) from an already-computed
qa_engine grounded result or multi_file_comparison Comparison Result;
nothing here performs a calculation of its own.

Kept deliberately small: the type vocabulary mirrors the existing Q&A
intent/comparison-type vocabulary rather than introducing a new taxonomy.
"""

EVIDENCE_TYPES = {
    "period_value",
    "period_extremum",
    "period_change",
    "column_stat",
    "anomaly_summary",
    "missing_values",
    "trend",
    "file_comparison",
    "forecast",
}


def make_evidence(evidence_id: str, evidence_type: str, source: dict, **fields) -> dict:
    """Build one Evidence dict. `source` identifies where the fact came
    from (file/column/period, as available) — never a `file_id`. Extra
    keyword fields (value, previous, current, absolute_change, ...) are
    stored verbatim; this function does not compute or transform them.
    """
    if not evidence_id or not str(evidence_id).strip():
        raise ValueError("evidence_id must be a non-empty string")
    if evidence_type not in EVIDENCE_TYPES:
        raise ValueError(f"Unknown evidence type: {evidence_type!r}")
    if source is not None and not isinstance(source, dict):
        raise TypeError("source must be a dict or None")
    if source and "file_id" in source:
        raise ValueError("Evidence source must not contain file_id")

    return {
        "evidence_id": str(evidence_id),
        "type": evidence_type,
        "source": dict(source or {}),
        **fields,
    }


def make_evidence_id(index: int) -> str:
    return f"ev_{index:03d}"
