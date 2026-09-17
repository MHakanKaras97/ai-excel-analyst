import json

from src.ai_provider import AIProviderError
from src.qa_interpreter import interpret_question

VALID_INTENT = {
    "intent": "period_value",
    "metric": None,
    "column_hint": "Revenue",
    "period_hint": "March",
    "from_period_hint": None,
    "to_period_hint": None,
}


class FakeProvider:
    def __init__(self, response=None, error: AIProviderError | None = None):
        self._response = response
        self._error = error
        self.last_prompt = None
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        if self._error is not None:
            raise self._error
        return self._response


def _response(**overrides) -> str:
    payload = dict(VALID_INTENT)
    payload.update(overrides)
    return json.dumps(payload)


def test_valid_structured_intent_is_accepted():
    provider = FakeProvider(response=_response())

    result = interpret_question("What was Revenue in March?", ["Revenue"], provider)

    assert result["is_valid"] is True
    assert result["intent"]["intent"] == "period_value"
    assert result["intent"]["column_hint"] == "Revenue"
    assert result["intent"]["period_hint"] == "March"


def test_build_qa_prompt_is_used():
    provider = FakeProvider(response=_response())

    interpret_question("What was Revenue in March?", ["Revenue", "Month"], provider)

    assert "What was Revenue in March?" in provider.last_prompt
    assert "Revenue" in provider.last_prompt
    assert "Month" in provider.last_prompt


def test_provider_generate_called_exactly_once():
    provider = FakeProvider(response=_response())

    interpret_question("Any question", [], provider)

    assert provider.call_count == 1


def test_malformed_json_is_rejected():
    provider = FakeProvider(response="this is not json")

    result = interpret_question("Any question", [], provider)

    assert result["is_valid"] is False
    assert result["reason"] == "invalid_json"


def test_non_object_json_is_rejected():
    provider = FakeProvider(response="[1, 2, 3]")

    result = interpret_question("Any question", [], provider)

    assert result["is_valid"] is False
    assert result["reason"] == "invalid_schema"


def test_missing_required_key_is_rejected():
    incomplete = dict(VALID_INTENT)
    del incomplete["period_hint"]
    provider = FakeProvider(response=json.dumps(incomplete))

    result = interpret_question("Any question", [], provider)

    assert result["is_valid"] is False
    assert result["reason"] == "invalid_schema"


def test_invalid_intent_is_rejected():
    provider = FakeProvider(response=_response(intent="delete_everything"))

    result = interpret_question("Any question", [], provider)

    assert result["is_valid"] is False
    assert result["reason"] == "invalid_schema"


def test_invalid_metric_is_rejected():
    provider = FakeProvider(response=_response(metric="average"))

    result = interpret_question("Any question", [], provider)

    assert result["is_valid"] is False
    assert result["reason"] == "invalid_schema"


def test_invalid_hint_type_is_rejected():
    provider = FakeProvider(response=_response(column_hint=42))

    result = interpret_question("Any question", [], provider)

    assert result["is_valid"] is False
    assert result["reason"] == "invalid_schema"


def test_extra_key_is_rejected():
    payload = dict(VALID_INTENT)
    payload["unexpected_key"] = "surprise"
    provider = FakeProvider(response=json.dumps(payload))

    result = interpret_question("Any question", [], provider)

    assert result["is_valid"] is False
    assert result["reason"] == "invalid_schema"


def test_provider_error_is_handled():
    provider = FakeProvider(error=AIProviderError("missing_api_key"))

    result = interpret_question("Any question", [], provider)

    assert result["is_valid"] is False
    assert result["reason"] == "missing_api_key"


def test_null_hints_are_accepted():
    provider = FakeProvider(response=_response(
        intent="missing_values",
        metric=None,
        column_hint=None,
        period_hint=None,
        from_period_hint=None,
        to_period_hint=None,
    ))

    result = interpret_question("Which columns have missing values?", [], provider)

    assert result["is_valid"] is True
    assert result["intent"]["column_hint"] is None
    assert result["intent"]["period_hint"] is None


def test_valid_period_change_with_from_and_to_hints_is_accepted():
    provider = FakeProvider(response=_response(
        intent="period_change",
        metric=None,
        column_hint="Revenue",
        period_hint=None,
        from_period_hint="March",
        to_period_hint="April",
    ))

    result = interpret_question("Compare March to April Revenue", ["Revenue"], provider)

    assert result["is_valid"] is True
    assert result["intent"]["from_period_hint"] == "March"
    assert result["intent"]["to_period_hint"] == "April"


def test_valid_period_extremum_with_min_and_max_is_accepted():
    for metric in ("min", "max"):
        provider = FakeProvider(response=_response(intent="period_extremum", metric=metric))

        result = interpret_question("What was the peak?", ["Revenue"], provider)

        assert result["is_valid"] is True
        assert result["intent"]["metric"] == metric


def test_unsupported_intent_is_accepted_as_a_valid_intent():
    provider = FakeProvider(response=_response(
        intent="unsupported",
        metric=None,
        column_hint=None,
        period_hint=None,
        from_period_hint=None,
        to_period_hint=None,
    ))

    result = interpret_question("What is the meaning of life?", [], provider)

    assert result["is_valid"] is True
    assert result["intent"]["intent"] == "unsupported"


def test_no_numeric_grounding_logic_is_introduced():
    provider = FakeProvider(response=_response(
        intent="column_stat",
        metric="sum",
        column_hint="an unprecedented 999999.99 figure",
    ))

    result = interpret_question("What is the sum?", ["Revenue"], provider)

    assert result["is_valid"] is True
    assert result["intent"]["column_hint"] == "an unprecedented 999999.99 figure"
