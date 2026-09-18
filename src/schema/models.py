"""Semantic schema model (V0.8.4).

A small, deliberately narrow vocabulary describing what a dataset column
*means*, layered on top of (never replacing) the existing dtype/profiling
information from data_profiler.py/data_normalizer.py. Kept intentionally
small — no large ontology, no unit-conversion table.
"""

COLUMN_ROLES = {"dimension", "measure", "date", "identifier", "unknown"}
SEMANTIC_TYPES = {"numeric", "currency", "percentage", "date", "text", "identifier", "unknown"}
TIME_ROLES = {"period", "timestamp", "none"}


def make_column_schema(name: str, role: str, semantic_type: str, unit, time_role,
                        confidence: float, evidence: dict) -> dict:
    """Build a ColumnSchema dict. Validates the closed vocabularies so an
    invalid role/semantic_type/time_role can never silently enter the
    schema, regardless of whether it came from deterministic analysis or an
    (unvalidated) LLM proposal.
    """
    if role not in COLUMN_ROLES:
        raise ValueError(f"Unknown column role: {role!r}")
    if semantic_type not in SEMANTIC_TYPES:
        raise ValueError(f"Unknown semantic_type: {semantic_type!r}")
    if time_role not in TIME_ROLES:
        raise ValueError(f"Unknown time_role: {time_role!r}")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence must be within [0, 1]: {confidence!r}")

    return {
        "name": str(name),
        "role": role,
        "semantic_type": semantic_type,
        "unit": unit,
        "time_role": time_role,
        "confidence": float(confidence),
        "evidence": evidence or {},
    }


def make_dataset_schema(columns: list) -> dict:
    return {"columns": list(columns)}
