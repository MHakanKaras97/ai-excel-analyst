import json

from src.ai_provider import AIProvider, AIProviderError
from src.chart_intent_prompt_builder import build_chart_intent_prompt

REQUIRED_KEYS = {
    "intent",
    "column_hint",
    "metric",
    "period_hint",
    "from_period_hint",
    "to_period_hint",
}

ALLOWED_INTENTS = {
    "trend_chart",
    "numeric_summary_chart",
    "period_change_chart",
    "missing_values_chart",
    "unsupported",
}

ALLOWED_METRICS = {"sum", "mean", "median", "min", "max", "std", "count", None}

STRING_OR_NULL_FIELDS = (
    "column_hint",
    "period_hint",
    "from_period_hint",
    "to_period_hint",
)


def _failure(reason: str) -> dict:
    return {"is_valid": False, "reason": reason}


def _validate_schema(parsed) -> bool:
    if not isinstance(parsed, dict):
        return False

    if set(parsed.keys()) != REQUIRED_KEYS:
        return False

    if parsed["intent"] not in ALLOWED_INTENTS:
        return False

    if parsed["metric"] not in ALLOWED_METRICS:
        return False

    for key in STRING_OR_NULL_FIELDS:
        value = parsed[key]
        if value is not None and not isinstance(value, str):
            return False

    return True


def interpret_chart_question(question: str, column_names: list[str], provider: AIProvider) -> dict:
    prompt = build_chart_intent_prompt(question, column_names)

    try:
        response_text = provider.generate(prompt)
    except AIProviderError as exc:
        return _failure(exc.reason)

    try:
        parsed = json.loads(response_text)
    except (json.JSONDecodeError, TypeError):
        return _failure("invalid_json")

    if not _validate_schema(parsed):
        return _failure("invalid_schema")

    return {"is_valid": True, "intent": parsed}
