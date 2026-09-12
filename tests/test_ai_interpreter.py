import json

from src.ai_interpreter import interpret
from src.ai_provider import AIProviderError

VALID_RESPONSE = json.dumps({
    "summary": "Revenue grew steadily across all quarters.",
    "key_insights": ["Q1 to Q4 shows consistent growth."],
    "trend_interpretation": "increasing",
    "warnings": [],
    "recommendations": ["Review Q3 for continued momentum."],
})


class FakeProvider:
    def __init__(self, response=None, error: AIProviderError | None = None):
        self._response = response
        self._error = error
        self.last_prompt = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        if self._error is not None:
            raise self._error
        return self._response


def test_interpret_valid_response():
    provider = FakeProvider(response=VALID_RESPONSE)

    result = interpret({"numeric_summary": {"columns": []}}, provider)

    assert result["is_valid"] is True
    assert result["reason"] is None
    assert result["insight"]["summary"] == "Revenue grew steadily across all quarters."
    assert result["insight"]["trend_interpretation"] == "increasing"
    assert result["insight"]["warnings"] == []


def test_interpret_passes_analytics_data_into_prompt():
    provider = FakeProvider(response=VALID_RESPONSE)
    payload = {"numeric_summary": {"columns": [{"name": "revenue", "sum": 999.0}]}}

    interpret(payload, provider)

    assert "revenue" in provider.last_prompt
    assert "999.0" in provider.last_prompt


def test_interpret_provider_error_passthrough():
    provider = FakeProvider(error=AIProviderError("missing_api_key"))

    result = interpret({}, provider)

    assert result["is_valid"] is False
    assert result["reason"] == "missing_api_key"
    assert result["insight"] is None


def test_interpret_provider_generic_error():
    provider = FakeProvider(error=AIProviderError("provider_error"))

    result = interpret({}, provider)

    assert result["reason"] == "provider_error"


def test_interpret_empty_response():
    provider = FakeProvider(response="")

    result = interpret({}, provider)

    assert result["is_valid"] is False
    assert result["reason"] == "empty_response"


def test_interpret_whitespace_only_response():
    provider = FakeProvider(response="   \n  ")

    result = interpret({}, provider)

    assert result["reason"] == "empty_response"


def test_interpret_non_json_response():
    provider = FakeProvider(response="this is not json")

    result = interpret({}, provider)

    assert result["is_valid"] is False
    assert result["reason"] == "malformed_output"


def test_interpret_json_missing_required_key():
    incomplete = json.dumps({"summary": "ok", "key_insights": []})
    provider = FakeProvider(response=incomplete)

    result = interpret({}, provider)

    assert result["reason"] == "malformed_output"


def test_interpret_json_wrong_field_type():
    wrong_type = json.dumps({
        "summary": "ok",
        "key_insights": "not a list",
        "trend_interpretation": None,
        "warnings": [],
        "recommendations": [],
    })
    provider = FakeProvider(response=wrong_type)

    result = interpret({}, provider)

    assert result["reason"] == "malformed_output"


def test_interpret_json_array_instead_of_object():
    provider = FakeProvider(response="[1, 2, 3]")

    result = interpret({}, provider)

    assert result["reason"] == "malformed_output"
