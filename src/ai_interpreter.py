import json

from src.ai_prompt_builder import build_prompt
from src.ai_provider import AIProvider, AIProviderError

REQUIRED_KEYS = {"summary", "key_insights", "trend_interpretation", "warnings", "recommendations"}
LIST_FIELDS = ("key_insights", "warnings", "recommendations")


def _failure(reason: str) -> dict:
    return {"is_valid": False, "reason": reason, "insight": None}


def _validate_shape(parsed) -> bool:
    if not isinstance(parsed, dict) or not REQUIRED_KEYS.issubset(parsed.keys()):
        return False

    if not isinstance(parsed["summary"], str):
        return False

    if parsed["trend_interpretation"] is not None and not isinstance(parsed["trend_interpretation"], str):
        return False

    return all(isinstance(parsed[field], list) for field in LIST_FIELDS)


def interpret(analytics_payload: dict, provider: AIProvider) -> dict:
    prompt = build_prompt(analytics_payload)

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
    }
