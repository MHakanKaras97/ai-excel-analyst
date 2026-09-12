import json
import re

from src.ai_prompt_builder import build_prompt
from src.ai_provider import AIProvider, AIProviderError

REQUIRED_KEYS = {"summary", "key_insights", "trend_interpretation", "warnings", "recommendations"}
LIST_FIELDS = ("key_insights", "warnings", "recommendations")

# Matches, in priority order: comma-grouped numbers with an optional decimal
# part (3,266,656.8), plain decimals (3266656.8), and plain integers (3266656).
NUMBER_PATTERN = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+\.\d+|-?\d+")

# Period labels like "2023-01" or "2023-01-15" (as produced by compare_periods /
# build_monthly_series) are dates, not quantities — strip them before number
# extraction so a year fragment (e.g. "2023") isn't mistaken for a cited metric.
DATE_LABEL_PATTERN = re.compile(r"\b\d{4}-\d{2}(?:-\d{2})?\b")

MEANINGFUL_INTEGER_THRESHOLD = 10


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


def _is_meaningful_number(value: float) -> bool:
    """Small integers (e.g. counts like "3 columns") are common in prose and
    are not worth checking; larger integers and any non-integer decimal are."""
    is_integer_valued = value == int(value)
    if is_integer_valued and abs(value) < MEANINGFUL_INTEGER_THRESHOLD:
        return False
    return True


def _collect_payload_numbers(value) -> set:
    numbers = set()
    if isinstance(value, dict):
        for item in value.values():
            numbers |= _collect_payload_numbers(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            numbers |= _collect_payload_numbers(item)
    elif isinstance(value, bool):
        pass
    elif isinstance(value, (int, float)):
        number = float(value)
        if _is_meaningful_number(number):
            numbers.add(round(number, 6))
    return numbers


def _extract_numbers_from_text(text: str) -> set:
    text = DATE_LABEL_PATTERN.sub(" ", text)
    numbers = set()
    for match in NUMBER_PATTERN.finditer(text):
        try:
            number = float(match.group().replace(",", ""))
        except ValueError:
            continue
        if _is_meaningful_number(number):
            numbers.add(round(number, 6))
    return numbers


def _response_text(parsed: dict) -> str:
    parts = [parsed["summary"]]
    if parsed["trend_interpretation"]:
        parts.append(str(parsed["trend_interpretation"]))
    for field in LIST_FIELDS:
        parts.extend(str(item) for item in parsed[field])
    return "\n".join(parts)


def _number_is_supported(number: float, payload_numbers: set) -> bool:
    for payload_number in payload_numbers:
        if abs(number - payload_number) < 1e-6:
            return True
        # Tolerate the model citing a rounded/truncated form of a payload value.
        if int(number) == int(payload_number):
            return True
    return False


def _has_hallucinated_numbers(parsed: dict, analytics_payload: dict) -> bool:
    payload_numbers = _collect_payload_numbers(analytics_payload)
    response_numbers = _extract_numbers_from_text(_response_text(parsed))

    return any(not _number_is_supported(number, payload_numbers) for number in response_numbers)


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

    if _has_hallucinated_numbers(parsed, analytics_payload):
        return _failure("hallucinated_numeric_value")

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
