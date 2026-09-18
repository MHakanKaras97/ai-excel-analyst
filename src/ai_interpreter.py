import json
import re
from typing import Callable

from src.ai_prompt_builder import build_prompt
from src.ai_provider import AIProvider, AIProviderError

REQUIRED_KEYS = {"summary", "key_insights", "trend_interpretation", "warnings", "recommendations"}
LIST_FIELDS = ("key_insights", "warnings", "recommendations")

# Matches, in priority order: comma-grouped numbers with an optional decimal
# part (3,266,656.8), plain decimals (3266656.8), and plain integers (3266656).
NUMBER_PATTERN = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+\.\d+|-?\d+")

# Period labels like "2023-01", "2023-01-15", or a full ISO timestamp such as
# "2023-01-15T00:00:00" (as produced by compare_periods / build_monthly_series /
# analytics_engine's isoformat() dates) are dates, not quantities — strip them
# before number extraction so a year/day fragment isn't mistaken for a cited
# metric. The trailing \b alone doesn't fire between a digit and a following
# letter (e.g. the "T" separator), so the day/time portion must be matched
# explicitly rather than relying on a boundary right after the date.
DATE_LABEL_PATTERN = re.compile(
    r"\b\d{4}-\d{2}(?:-\d{2}(?:[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?)?)?\b"
)

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


def _collect_payload_labels(value) -> set:
    """Collect string leaf values from the payload that mix letters and
    digits (e.g. "Budget_2024", "Revenue2024") — deterministic labels whose
    embedded digits are clearly part of the label's identity, not a
    standalone numeric claim. Purely numeric strings (e.g. a "2024-03"
    period label) are excluded here since they're already handled by
    DATE_LABEL_PATTERN, and treating a bare number as a maskable "label"
    would let it swallow a genuinely different numeric claim that happens
    to reuse the same digits.
    """
    labels = set()
    if isinstance(value, dict):
        for item in value.values():
            labels |= _collect_payload_labels(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            labels |= _collect_payload_labels(item)
    elif isinstance(value, str):
        if any(ch.isdigit() for ch in value) and any(ch.isalpha() for ch in value):
            labels.add(value)
    return labels


def _mask_known_labels(text: str, known_labels: set) -> str:
    """Remove verbatim occurrences of known digit-bearing payload labels
    from `text` before number extraction, so a response merely echoing a
    supplied label (e.g. "Budget_2024") doesn't have its embedded digits
    mistaken for a standalone numeric claim. Only exact, literal
    occurrences of a known label are masked — any other digit sequence
    (including the same digits appearing outside that exact label) is left
    for normal number extraction and must still be a supported fact.
    """
    for label in sorted(known_labels, key=len, reverse=True):
        text = text.replace(label, " ")
    return text


def _extract_numbers_from_text(text: str, known_labels: set = frozenset()) -> set:
    text = _mask_known_labels(text, known_labels)
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
    payload_labels = _collect_payload_labels(analytics_payload)
    response_numbers = _extract_numbers_from_text(_response_text(parsed), payload_labels)

    return any(not _number_is_supported(number, payload_numbers) for number in response_numbers)


# Public alias: reused by src/evidence_ai_interpreter.py (V0.8.14) so the
# evidence-aware AI layer doesn't reimplement numeric grounding — `parsed`
# only needs the same 5 prose fields (summary/key_insights/
# trend_interpretation/warnings/recommendations) checked here; any extra
# keys in a superset schema are ignored.
has_hallucinated_numbers = _has_hallucinated_numbers


def interpret(analytics_payload: dict, provider: AIProvider,
               prompt_builder: Callable[[dict], str] = build_prompt) -> dict:
    prompt = prompt_builder(analytics_payload)

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
