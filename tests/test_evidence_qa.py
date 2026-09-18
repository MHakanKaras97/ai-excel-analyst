import json

from src.ai_provider import AIProviderError
from src.evidence_qa import answer_comparison_with_evidence, answer_question_with_evidence


class ScriptedProvider:
    """Returns successive canned responses, one per call, in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        response = self._responses[self.call_count]
        self.call_count += 1
        if isinstance(response, AIProviderError):
            raise response
        return response


def _intent_response(**overrides):
    payload = {"intent": "column_stat", "metric": "sum", "column_hint": "Revenue",
               "period_hint": None, "from_period_hint": None, "to_period_hint": None}
    payload.update(overrides)
    return json.dumps(payload)


def _explanation_response(**overrides):
    payload = {
        "summary": "The sum of Revenue is 460000.0.", "key_insights": [], "trend_interpretation": None,
        "warnings": [], "recommendations": [], "evidence_ids_used": ["ev_001"],
    }
    payload.update(overrides)
    return json.dumps(payload)


ANALYSIS_PAYLOAD = {
    "numeric_summary": {"columns": [{"name": "Revenue", "sum": 460000.0, "mean": 115000.0,
                                      "median": 115000.0, "min": 100000.0, "max": 130000.0,
                                      "std": 12909.9, "count": 4, "missing_count": 0}]},
    "profile": {"columns": [{"name": "Revenue"}]},
}


def test_valid_question_produces_deterministic_answer_and_ai_explanation():
    provider = ScriptedProvider([_intent_response(), _explanation_response()])

    result = answer_question_with_evidence("What is the total revenue?", ["Revenue"], provider, ANALYSIS_PAYLOAD)

    assert result["deterministic_answer"] == "The sum of Revenue is 460000.0."
    assert result["evidence"][0]["evidence_id"] == "ev_001"
    assert result["ai_explanation"]["is_valid"] is True


def test_unsupported_question_has_no_evidence_and_deterministic_answer_still_produced():
    provider = ScriptedProvider([_intent_response(intent="unsupported", column_hint=None, metric=None)])

    result = answer_question_with_evidence("What's the weather?", ["Revenue"], provider, ANALYSIS_PAYLOAD)

    assert result["deterministic_answer"] == "That question isn't supported yet."
    assert result["evidence"] == []
    assert result["ai_explanation"] is None
    assert provider.call_count == 1  # no second (explanation) call was made


def test_ai_explanation_failure_does_not_affect_deterministic_answer():
    provider = ScriptedProvider([_intent_response(), "not json"])

    result = answer_question_with_evidence("What is the total revenue?", ["Revenue"], provider, ANALYSIS_PAYLOAD)

    assert result["deterministic_answer"] == "The sum of Revenue is 460000.0."
    assert result["ai_explanation"]["is_valid"] is False


def test_provider_error_on_interpretation_still_returns_deterministic_failure_message():
    provider = ScriptedProvider([AIProviderError("provider_error")])

    result = answer_question_with_evidence("What is the total revenue?", ["Revenue"], provider, ANALYSIS_PAYLOAD)

    assert "AI provider returned an error" in result["deterministic_answer"]
    assert result["evidence"] == []
    assert result["ai_explanation"] is None


def test_column_not_found_produces_no_evidence():
    provider = ScriptedProvider([_intent_response(column_hint="Nonexistent")])

    result = answer_question_with_evidence("total nonexistent?", ["Revenue"], provider, ANALYSIS_PAYLOAD)

    assert result["evidence"] == []
    assert result["ai_explanation"] is None
    assert "couldn't find a column" in result["deterministic_answer"]


# ==================================================
# answer_comparison_with_evidence
# ==================================================


def _comparison_intent_response(**overrides):
    payload = {
        "intent": "file_value_comparison", "column_hint": "Revenue", "metric": "sum",
        "period_hint": None, "from_file_hint": "previous", "to_file_hint": "current",
    }
    payload.update(overrides)
    return json.dumps(payload)


def _data_files():
    return {
        "file-a": {
            "file_id": "file-a", "filename": "jan.xlsx", "display_name": "January", "role": "previous",
            "analysis": {"numeric_summary": {"columns": [{"name": "Revenue", "sum": 210.0, "count": 2}]}},
        },
        "file-b": {
            "file_id": "file-b", "filename": "feb.xlsx", "display_name": "February", "role": "current",
            "analysis": {"numeric_summary": {"columns": [{"name": "Revenue", "sum": 410.0, "count": 2}]}},
        },
    }


def test_comparison_question_produces_deterministic_answer_and_evidence():
    provider = ScriptedProvider([_comparison_intent_response(), _explanation_response(
        summary="Revenue for February is 410.0, compared with 210.0 for January.",
    )])

    result = answer_comparison_with_evidence(
        "How does revenue compare?", ["Revenue"], ["previous", "current"], provider, _data_files(),
    )

    assert "410.0" in result["deterministic_answer"]
    assert result["evidence"][0]["type"] == "file_comparison"
    assert "file_id" not in str(result["evidence"])
    assert result["ai_explanation"]["is_valid"] is True


def test_comparison_missing_role_produces_no_evidence():
    provider = ScriptedProvider([_comparison_intent_response()])
    data_files = _data_files()
    data_files["file-a"]["role"] = None
    data_files["file-b"]["role"] = None

    result = answer_comparison_with_evidence(
        "How does revenue compare?", ["Revenue"], [], provider, data_files,
    )

    assert result["evidence"] == []
    assert result["ai_explanation"] is None
