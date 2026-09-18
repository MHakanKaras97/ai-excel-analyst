import json

from src.ai_provider import AIProviderError
from src.multi_file_qa_interpreter import interpret_multi_file_question

VALID_INTENT = {
    "intent": "file_value_comparison", "column_hint": "Revenue", "metric": "sum",
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


def test_valid_file_value_comparison_is_accepted():
    provider = FakeProvider(response=_response())

    result = interpret_multi_file_question("Compare revenue", ["Revenue"], ["previous", "current"], provider)

    assert result["is_valid"] is True
    assert result["intent"]["intent"] == "file_value_comparison"
    assert result["intent"]["from_file_hint"] == "previous"
    assert result["intent"]["to_file_hint"] == "current"


def test_valid_period_comparison_is_accepted():
    provider = FakeProvider(response=_response(intent="file_period_comparison", period_hint="March", metric=None))

    result = interpret_multi_file_question("Compare March", [], [], provider)

    assert result["is_valid"] is True
    assert result["intent"]["period_hint"] == "March"


def test_valid_trend_comparison_is_accepted():
    provider = FakeProvider(response=_response(intent="file_trend_comparison", column_hint=None, metric=None))

    result = interpret_multi_file_question("Compare trends", [], [], provider)

    assert result["is_valid"] is True
    assert result["intent"]["intent"] == "file_trend_comparison"


def test_valid_anomaly_comparison_is_accepted():
    provider = FakeProvider(response=_response(intent="file_anomaly_comparison", column_hint=None, metric=None))

    result = interpret_multi_file_question("Compare anomalies", [], [], provider)

    assert result["is_valid"] is True
    assert result["intent"]["intent"] == "file_anomaly_comparison"


def test_valid_unsupported_is_accepted():
    provider = FakeProvider(response=_response(
        intent="unsupported", column_hint=None, metric=None, from_file_hint=None, to_file_hint=None,
    ))

    result = interpret_multi_file_question("What's the weather?", [], [], provider)

    assert result["is_valid"] is True
    assert result["intent"]["intent"] == "unsupported"


def test_all_six_fields_present_in_successful_result():
    provider = FakeProvider(response=_response())

    result = interpret_multi_file_question("q", [], [], provider)

    assert set(result["intent"].keys()) == {
        "intent", "column_hint", "metric", "period_hint", "from_file_hint", "to_file_hint",
    }


def test_extra_key_is_rejected():
    payload = dict(VALID_INTENT)
    payload["unexpected"] = "x"
    provider = FakeProvider(response=json.dumps(payload))

    result = interpret_multi_file_question("q", [], [], provider)

    assert result["is_valid"] is False and result["reason"] == "invalid_schema"


def test_missing_key_is_rejected():
    incomplete = dict(VALID_INTENT)
    del incomplete["period_hint"]
    provider = FakeProvider(response=json.dumps(incomplete))

    result = interpret_multi_file_question("q", [], [], provider)

    assert result["is_valid"] is False and result["reason"] == "invalid_schema"


def test_invalid_intent_is_rejected():
    provider = FakeProvider(response=_response(intent="file_forecast_comparison"))

    result = interpret_multi_file_question("q", [], [], provider)

    assert result["is_valid"] is False and result["reason"] == "invalid_schema"


def test_invalid_metric_is_rejected():
    provider = FakeProvider(response=_response(metric="average"))

    result = interpret_multi_file_question("q", [], [], provider)

    assert result["is_valid"] is False and result["reason"] == "invalid_schema"


def test_numeric_hint_type_is_rejected():
    provider = FakeProvider(response=_response(from_file_hint=42))

    result = interpret_multi_file_question("q", [], [], provider)

    assert result["is_valid"] is False and result["reason"] == "invalid_schema"


def test_malformed_json_is_rejected():
    provider = FakeProvider(response="not json")

    result = interpret_multi_file_question("q", [], [], provider)

    assert result["is_valid"] is False and result["reason"] == "invalid_json"


def test_provider_error_is_handled():
    provider = FakeProvider(error=AIProviderError("missing_api_key"))

    result = interpret_multi_file_question("q", [], [], provider)

    assert result["is_valid"] is False and result["reason"] == "missing_api_key"


def test_provider_generate_called_exactly_once():
    provider = FakeProvider(response=_response())

    interpret_multi_file_question("q", ["Revenue"], ["previous"], provider)

    assert provider.call_count == 1


def test_prompt_receives_expected_question_columns_and_files():
    provider = FakeProvider(response=_response())

    interpret_multi_file_question("Compare revenue", ["Revenue"], ["previous", "current"], provider)

    assert "Compare revenue" in provider.last_prompt
    assert "Revenue" in provider.last_prompt
    assert "previous" in provider.last_prompt


def test_does_not_mutate_inputs():
    provider = FakeProvider(response=_response())
    columns, files = ["Revenue"], ["previous"]
    before_columns, before_files = list(columns), list(files)

    interpret_multi_file_question("q", columns, files, provider)

    assert columns == before_columns and files == before_files
