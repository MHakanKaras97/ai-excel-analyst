"""Closed-schema interpreter for LLM semantic column proposals (V0.8.13).

Mirrors the existing qa_interpreter.py / multi_file_qa_interpreter.py
pattern: exactly one LLM call, a fixed set of allowed keys/enum values, and
a hard rejection (never a best-effort guess) for anything else. The LLM
output is a proposal only — schema_resolver.merge_llm_schema_proposal()
performs the actual (Python-authoritative) validation against each
column's deterministic evidence.
"""
import json

from src.ai_provider import AIProvider, AIProviderError
from src.schema.models import COLUMN_ROLES, SEMANTIC_TYPES, TIME_ROLES
from src.schema.semantic_prompt_builder import build_semantic_schema_prompt

REQUIRED_PROPOSAL_KEYS = {"name", "role", "semantic_type", "unit", "time_role"}


def _failure(reason: str) -> dict:
    return {"is_valid": False, "reason": reason, "proposals": None}


def _valid_proposal_entry(entry, expected_names: set) -> bool:
    if not isinstance(entry, dict) or set(entry.keys()) != REQUIRED_PROPOSAL_KEYS:
        return False
    if entry["name"] not in expected_names:
        return False
    if entry["role"] not in COLUMN_ROLES:
        return False
    if entry["semantic_type"] not in SEMANTIC_TYPES:
        return False
    if entry["unit"] is not None and not isinstance(entry["unit"], str):
        return False
    if entry["time_role"] not in TIME_ROLES:
        return False
    return True


def interpret_semantic_schema(candidate_columns: list[dict], provider: AIProvider,
                               prompt_builder=build_semantic_schema_prompt) -> dict:
    """`candidate_columns`: compact per-column metadata dicts (name, dtype,
    inferred_type, sample_values, candidate_role, candidate_semantic_type,
    candidate_confidence) — never the raw dataset, never a file_id.

    Returns {"is_valid": True, "reason": None, "proposals": {name: {role,
    semantic_type, unit, time_role}}} on success, or an invalid result with
    a deterministic failure reason. A proposal need not cover every column
    (a partial response is accepted); any entry that fails validation, or
    any duplicate proposal for the same column name, invalidates the whole
    response rather than silently keeping the well-formed parts.
    """
    prompt = prompt_builder(candidate_columns)

    try:
        response_text = provider.generate(prompt)
    except AIProviderError as exc:
        return _failure(exc.reason)

    if not response_text or not response_text.strip():
        return _failure("empty_response")

    try:
        parsed = json.loads(response_text)
    except (json.JSONDecodeError, TypeError):
        return _failure("invalid_json")

    if not isinstance(parsed, dict) or set(parsed.keys()) != {"columns"} or not isinstance(parsed["columns"], list):
        return _failure("invalid_schema")

    expected_names = {c["name"] for c in candidate_columns}
    proposals = {}
    for entry in parsed["columns"]:
        if not _valid_proposal_entry(entry, expected_names) or entry["name"] in proposals:
            return _failure("invalid_schema")
        proposals[entry["name"]] = {
            "role": entry["role"],
            "semantic_type": entry["semantic_type"],
            "unit": entry["unit"],
            "time_role": entry["time_role"],
        }

    return {"is_valid": True, "reason": None, "proposals": proposals}
