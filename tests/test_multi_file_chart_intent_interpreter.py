import json

from src.ai_provider import AIProviderError
from src.multi_file_chart_intent_interpreter import interpret_multi_file_chart_question
from src.multi_file_chart_intent_prompt_builder import build_multi_file_chart_intent_prompt

VALID_INTENT = {
    "intent": "file_comparison_chart", "column_hint": "Revenue", "metric": "sum",
    "period_hint": None, "from_file_hint": "previous", "to_file_hint": "current",
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


def test_prompt_builder_returns_string_with_expected_content():
    prompt = build_multi_file_chart_intent_prompt("Compare totals", ["Revenue"], ["previous", "current"])

    assert isinstance(prompt, str)
    assert "file_comparison_chart" in prompt
    assert "Compare totals" in prompt
    assert "Revenue" in prompt
    assert "previous" in prompt


def test_valid_file_comparison_chart_is_accepted():
    provider = FakeProvider(response=_response())

    result = interpret_multi_file_chart_question("Compare totals", ["Revenue"], ["previous", "current"], provider)

    assert result["is_valid"] is True
    assert result["intent"]["intent"] == "file_comparison_chart"


def test_valid_unsupported_is_accepted():
    provider = FakeProvider(response=_response(
        intent="unsupported", column_hint=None, metric=None, from_file_hint=None, to_file_hint=None,
    ))

    result = interpret_multi_file_chart_question("Show a trend chart across files", [], [], provider)

    assert result["is_valid"] is True
    assert result["intent"]["intent"] == "unsupported"


def test_invalid_intent_is_rejected():
    provider = FakeProvider(response=_response(intent="file_trend_comparison"))

    result = interpret_multi_file_chart_question("q", [], [], provider)

    assert result["is_valid"] is False and result["reason"] == "invalid_schema"


def test_extra_key_is_rejected():
    payload = dict(VALID_INTENT)
    payload["extra"] = 1
    provider = FakeProvider(response=json.dumps(payload))

    result = interpret_multi_file_chart_question("q", [], [], provider)

    assert result["is_valid"] is False and result["reason"] == "invalid_schema"


def test_malformed_json_is_rejected():
    provider = FakeProvider(response="not json")

    result = interpret_multi_file_chart_question("q", [], [], provider)

    assert result["is_valid"] is False and result["reason"] == "invalid_json"


def test_provider_error_is_handled():
    provider = FakeProvider(error=AIProviderError("provider_error"))

    result = interpret_multi_file_chart_question("q", [], [], provider)

    assert result["is_valid"] is False and result["reason"] == "provider_error"


def test_provider_generate_called_exactly_once():
    provider = FakeProvider(response=_response())

    interpret_multi_file_chart_question("q", ["Revenue"], ["previous"], provider)

    assert provider.call_count == 1


def test_does_not_mutate_inputs():
    provider = FakeProvider(response=_response())
    columns, files = ["Revenue"], ["previous"]
    before_columns, before_files = list(columns), list(files)

    interpret_multi_file_chart_question("q", columns, files, provider)

    assert columns == before_columns and files == before_files
