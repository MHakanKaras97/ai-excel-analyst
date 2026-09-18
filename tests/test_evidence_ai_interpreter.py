import json

from src.ai_provider import AIProviderError
from src.evidence_ai_interpreter import interpret_with_evidence
from src.evidence_prompt_builder import build_evidence_prompt


class FakeProvider:
    def __init__(self, response=None, error: AIProviderError | None = None):
        self._response = response
        self._error = error
        self.call_count = 0
        self.last_prompt = None

    def generate(self, prompt: str) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        if self._error is not None:
            raise self._error
        return self._response


EVIDENCE = [
    {"evidence_id": "ev_001", "type": "period_change", "source": {"column": "Revenue"},
     "previous": 150000, "current": 125000, "absolute_change": -25000, "percentage_change": -16.67},
]


def _valid_response(**overrides) -> str:
    payload = {
        "summary": "Revenue decreased from 150000 to 125000.",
        "key_insights": ["Revenue changed from 150000 to 125000, a change of -25000 (-16.67%)."],
        "trend_interpretation": None,
        "warnings": [],
        "recommendations": [],
        "evidence_ids_used": ["ev_001"],
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_prompt_builder_includes_evidence_and_question():
    prompt = build_evidence_prompt({"evidence": EVIDENCE, "question": "Why did revenue drop?"})

    assert "ev_001" in prompt
    assert "Why did revenue drop?" in prompt


def test_valid_response_grounded_in_evidence_is_accepted():
    provider = FakeProvider(response=_valid_response())

    result = interpret_with_evidence(EVIDENCE, provider)

    assert result["is_valid"] is True
    assert result["evidence_ids_used"] == ["ev_001"]
    assert result["insight"]["summary"].startswith("Revenue decreased")


def test_response_citing_unsupported_evidence_id_is_rejected():
    provider = FakeProvider(response=_valid_response(evidence_ids_used=["ev_999"]))

    result = interpret_with_evidence(EVIDENCE, provider)

    assert result["is_valid"] is False
    assert result["reason"] == "unsupported_evidence"


def test_response_with_invented_number_is_rejected():
    provider = FakeProvider(response=_valid_response(summary="Revenue decreased to 999999."))

    result = interpret_with_evidence(EVIDENCE, provider)

    assert result["is_valid"] is False
    assert result["reason"] == "hallucinated_numeric_value"


def test_response_with_causal_claim_is_rejected():
    provider = FakeProvider(response=_valid_response(
        key_insights=["The decrease in units caused revenue to fall."],
    ))

    result = interpret_with_evidence(EVIDENCE, provider)

    assert result["is_valid"] is False
    assert result["reason"] == "unsupported_causal_claim"


def test_response_reusing_only_supported_numbers_and_coincidence_language_is_accepted():
    evidence = EVIDENCE + [
        {"evidence_id": "ev_002", "type": "file_comparison", "source": {"column": "Units"},
         "previous": 500, "current": 400, "absolute_change": -100, "percentage_change": -20.0},
    ]
    provider = FakeProvider(response=_valid_response(
        key_insights=["Units also decreased during the same period, from 500 to 400."],
        evidence_ids_used=["ev_001", "ev_002"],
    ))

    result = interpret_with_evidence(evidence, provider)

    assert result["is_valid"] is True


def test_malformed_json_is_rejected():
    provider = FakeProvider(response="not json")

    result = interpret_with_evidence(EVIDENCE, provider)

    assert result["is_valid"] is False and result["reason"] == "malformed_output"


def test_missing_evidence_ids_used_field_is_rejected():
    payload = {
        "summary": "x", "key_insights": [], "trend_interpretation": None,
        "warnings": [], "recommendations": [],
    }
    provider = FakeProvider(response=json.dumps(payload))

    result = interpret_with_evidence(EVIDENCE, provider)

    assert result["is_valid"] is False and result["reason"] == "malformed_output"


def test_extra_key_is_rejected():
    payload = json.loads(_valid_response())
    payload["extra"] = 1
    provider = FakeProvider(response=json.dumps(payload))

    result = interpret_with_evidence(EVIDENCE, provider)

    assert result["is_valid"] is False and result["reason"] == "malformed_output"


def test_provider_error_is_handled():
    provider = FakeProvider(error=AIProviderError("provider_error"))

    result = interpret_with_evidence(EVIDENCE, provider)

    assert result["is_valid"] is False and result["reason"] == "provider_error"


def test_provider_generate_called_exactly_once():
    provider = FakeProvider(response=_valid_response())

    interpret_with_evidence(EVIDENCE, provider)

    assert provider.call_count == 1


def test_forecast_evidence_worded_as_prediction_is_accepted():
    forecast_evidence = [
        {"evidence_id": "ev_001", "type": "forecast", "source": {"column": "Revenue", "period": "2026-04"},
         "forecast": 142000, "method": "naive"},
    ]
    provider = FakeProvider(response=_valid_response(
        summary="The baseline forecast for Revenue in 2026-04 is 142000.",
        key_insights=[],
    ))

    result = interpret_with_evidence(forecast_evidence, provider)

    assert result["is_valid"] is True


def test_does_not_mutate_evidence_list():
    before = [dict(e) for e in EVIDENCE]
    provider = FakeProvider(response=_valid_response())

    interpret_with_evidence(EVIDENCE, provider)

    assert EVIDENCE == before
