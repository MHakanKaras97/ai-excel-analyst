"""Evidence-aware AI interpretation (V0.8.14).

Sibling to ai_interpreter.py — same one-LLM-call, closed-schema, grounded
pattern — extended with two checks ai_interpreter.py's plain insight schema
has no field for: that every cited evidence_id actually exists
(src/evidence/validator.py), and that the response contains no unsupported
causal claim. Existing numeric-hallucination grounding is reused verbatim
via ai_interpreter.has_hallucinated_numbers() rather than reimplemented —
this module adds evidence-specific checks on top, it does not weaken or
duplicate the existing ones.
"""
import json

from src.ai_interpreter import has_hallucinated_numbers
from src.ai_provider import AIProvider, AIProviderError
from src.evidence.validator import contains_unsupported_causal_claim, validate_referenced_evidence_ids
from src.evidence_prompt_builder import build_evidence_prompt

REQUIRED_KEYS = {"summary", "key_insights", "trend_interpretation", "warnings", "recommendations", "evidence_ids_used"}
LIST_FIELDS = ("key_insights", "warnings", "recommendations", "evidence_ids_used")


def _failure(reason: str) -> dict:
    return {"is_valid": False, "reason": reason, "insight": None}


def _validate_shape(parsed) -> bool:
    if not isinstance(parsed, dict) or set(parsed.keys()) != REQUIRED_KEYS:
        return False
    if not isinstance(parsed["summary"], str):
        return False
    if parsed["trend_interpretation"] is not None and not isinstance(parsed["trend_interpretation"], str):
        return False
    if not all(isinstance(parsed[field], list) for field in LIST_FIELDS):
        return False
    return all(isinstance(item, str) for item in parsed["evidence_ids_used"])


def _joined_text(parsed: dict) -> str:
    parts = [parsed["summary"]]
    if parsed["trend_interpretation"]:
        parts.append(str(parsed["trend_interpretation"]))
    for field in ("key_insights", "warnings", "recommendations"):
        parts.extend(str(item) for item in parsed[field])
    return "\n".join(parts)


def interpret_with_evidence(evidence_list: list[dict], provider: AIProvider, question: str | None = None,
                             prompt_builder=build_evidence_prompt) -> dict:
    """`evidence_list`: Evidence dicts (src/evidence/builder.py). `question`
    is optional context for the Evidence-Aware Q&A flow (V0.8.15); omit for
    the Evidence-Aware AI Insight flow (V0.8.14), which has no question.
    """
    payload = {"evidence": evidence_list, "question": question}
    prompt = prompt_builder(payload)

    try:
        response_text = provider.generate(prompt)
    except AIProviderError as exc:
        return _failure(exc.reason)

    if not response_text or not response_text.strip():
        return _failure("empty_response")

    try:
        parsed = json.loads(response_text)
    except (json.JSONDecodeError, TypeError):
        return _failure("malformed_output")

    if not _validate_shape(parsed):
        return _failure("malformed_output")

    if has_hallucinated_numbers(parsed, payload):
        return _failure("hallucinated_numeric_value")

    reference_check = validate_referenced_evidence_ids(parsed["evidence_ids_used"], evidence_list)
    if not reference_check["valid"]:
        return _failure("unsupported_evidence")

    if contains_unsupported_causal_claim(_joined_text(parsed)):
        return _failure("unsupported_causal_claim")

    return {
        "is_valid": True,
        "reason": None,
        "insight": {
            "summary": parsed["summary"],
            "key_insights": parsed["key_insights"],
            "trend_interpretation": parsed["trend_interpretation"],
            "warnings": parsed["warnings"],
            "recommendations": parsed["recommendations"],
        },
        "evidence_ids_used": parsed["evidence_ids_used"],
    }
