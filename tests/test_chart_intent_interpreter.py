import json

from src.ai_provider import AIProviderError
from src.chart_intent_interpreter import interpret_chart_question

VALID_INTENT = {
    "intent": "trend_chart",
    "column_hint": "TotalPrice",
    "metric": None,
    "period_hint": None,
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


def test_valid_trend_chart_is_accepted():
    provider = FakeProvider(response=_response())

    result = interpret_chart_question("Show the TotalPrice trend.", ["TotalPrice"], provider)

    assert result["is_valid"] is True
    assert result["intent"]["intent"] == "trend_chart"
    assert result["intent"]["column_hint"] == "TotalPrice"


def test_valid_numeric_summary_chart_with_mean_is_accepted():
    provider = FakeProvider(response=_response(intent="numeric_summary_chart", metric="mean"))

    result = interpret_chart_question("Show the average TotalPrice.", ["TotalPrice"], provider)

    assert result["is_valid"] is True
    assert result["intent"]["intent"] == "numeric_summary_chart"
    assert result["intent"]["metric"] == "mean"


def test_valid_period_change_chart_is_accepted():
    provider = FakeProvider(response=_response(
        intent="period_change_chart", column_hint="TotalPrice",
        from_period_hint="January", to_period_hint="March",
    ))

    result = interpret_chart_question("Plot the change from January to March.", ["TotalPrice"], provider)

    assert result["is_valid"] is True
    assert result["intent"]["intent"] == "period_change_chart"
    assert result["intent"]["from_period_hint"] == "January"
    assert result["intent"]["to_period_hint"] == "March"


def test_valid_missing_values_chart_is_accepted():
    provider = FakeProvider(response=_response(
        intent="missing_values_chart", column_hint=None,
    ))

    result = interpret_chart_question("Show missing values.", [], provider)

    assert result["is_valid"] is True
    assert result["intent"]["intent"] == "missing_values_chart"


def test_valid_unsupported_is_accepted():
    provider = FakeProvider(response=_response(intent="unsupported", column_hint=None))

    result = interpret_chart_question("Predict next month's revenue.", [], provider)

    assert result["is_valid"] is True
    assert result["intent"]["intent"] == "unsupported"


def test_null_hints_are_accepted():
    provider = FakeProvider(response=_response(
        column_hint=None, metric=None, period_hint=None,
        from_period_hint=None, to_period_hint=None,
    ))

    result = interpret_chart_question("Show the trend.", [], provider)

    assert result["is_valid"] is True
    assert result["intent"]["column_hint"] is None
    assert result["intent"]["period_hint"] is None


def test_all_six_fields_present_in_successful_result():
    provider = FakeProvider(response=_response())

    result = interpret_chart_question("Show the TotalPrice trend.", ["TotalPrice"], provider)

    assert set(result["intent"].keys()) == {
        "intent", "column_hint", "metric", "period_hint", "from_period_hint", "to_period_hint",
    }


def test_extra_key_is_rejected():
    payload = dict(VALID_INTENT)
    payload["unexpected_key"] = "surprise"
    provider = FakeProvider(response=json.dumps(payload))

    result = interpret_chart_question("Show the trend.", [], provider)

    assert result["is_valid"] is False
    assert result["reason"] == "invalid_schema"


def test_missing_key_is_rejected():
    incomplete = dict(VALID_INTENT)
    del incomplete["period_hint"]
    provider = FakeProvider(response=json.dumps(incomplete))

    result = interpret_chart_question("Show the trend.", [], provider)

    assert result["is_valid"] is False
    assert result["reason"] == "invalid_schema"


def test_invalid_intent_is_rejected():
    provider = FakeProvider(response=_response(intent="forecast_chart"))

    result = interpret_chart_question("Predict next month.", [], provider)

    assert result["is_valid"] is False
    assert result["reason"] == "invalid_schema"


def test_invalid_metric_is_rejected():
    provider = FakeProvider(response=_response(metric="average"))

    result = interpret_chart_question("Show the average.", [], provider)

    assert result["is_valid"] is False
    assert result["reason"] == "invalid_schema"


def test_numeric_hint_type_is_rejected():
    provider = FakeProvider(response=_response(column_hint=42))

    result = interpret_chart_question("Show the trend.", [], provider)

    assert result["is_valid"] is False
    assert result["reason"] == "invalid_schema"


def test_malformed_json_is_rejected():
    provider = FakeProvider(response="this is not json")

    result = interpret_chart_question("Show the trend.", [], provider)

    assert result["is_valid"] is False
    assert result["reason"] == "invalid_json"


def test_empty_response_is_rejected():
    provider = FakeProvider(response="")

    result = interpret_chart_question("Show the trend.", [], provider)

    assert result["is_valid"] is False
    assert result["reason"] == "invalid_json"


def test_provider_error_is_handled():
    provider = FakeProvider(error=AIProviderError("missing_api_key"))

    result = interpret_chart_question("Show the trend.", [], provider)

    assert result["is_valid"] is False
    assert result["reason"] == "missing_api_key"


def test_provider_generate_called_exactly_once():
    provider = FakeProvider(response=_response())

    interpret_chart_question("Show the TotalPrice trend.", ["TotalPrice"], provider)

    assert provider.call_count == 1


def test_prompt_builder_receives_expected_question_and_columns():
    provider = FakeProvider(response=_response())

    interpret_chart_question("Show the TotalPrice trend.", ["TotalPrice", "Month"], provider)

    assert "Show the TotalPrice trend." in provider.last_prompt
    assert "TotalPrice" in provider.last_prompt
    assert "Month" in provider.last_prompt


def test_does_not_mutate_column_names_input():
    provider = FakeProvider(response=_response())
    columns = ["TotalPrice", "Month"]
    before = list(columns)

    interpret_chart_question("Show the TotalPrice trend.", columns, provider)

    assert columns == before
