"""V0.7.4 tests: interpret()'s optional prompt_builder parameter, the
comparison payload/prompt integration, and the label-aware grounding
enhancement that prevents digits embedded in known string labels (e.g.
"Budget_2024") from being mistaken for a hallucinated numeric claim.

Existing tests/test_ai_interpreter.py is left completely unmodified — this
file only adds new, comparison-specific coverage.
"""
import inspect
import json

from src.ai_interpreter import interpret
from src.ai_prompt_builder import build_prompt
from src.ai_provider import AIProviderError
from src.comparison_ai_payload import build_comparison_ai_payload
from src.comparison_prompt_builder import build_comparison_prompt


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


def _insight_response(summary="A comparison summary.", key_insights=None, trend_interpretation=None,
                       warnings=None, recommendations=None):
    return json.dumps({
        "summary": summary,
        "key_insights": key_insights or [],
        "trend_interpretation": trend_interpretation,
        "warnings": warnings or [],
        "recommendations": recommendations or [],
    })


def _comparison_result(**overrides):
    result = {
        "found": True, "reason": None, "comparison_type": "file_value_comparison",
        "metric": "sum", "column": "TotalPrice", "period": None,
        "file_a": {"file_id": "f3a1c9e2b8d74a1f9c3e5b7a2d4f6081", "display_name": "Budget_2024", "role": "previous"},
        "file_b": {"file_id": "9bd27a41e6c34f2ab1d0987654321abc", "display_name": "Budget_2025", "role": "current"},
        "previous": 100000.0, "current": 130000.0, "absolute_change": 30000.0,
        "percentage_change": 30.0, "is_valid": True, "extra": {},
    }
    result.update(overrides)
    return result


def _ai_payload(**overrides):
    return build_comparison_ai_payload(_comparison_result(**overrides))


# ==================================================
# C. Backward compatibility
# ==================================================


def test_interpret_two_argument_call_behaves_exactly_as_before():
    provider = FakeProvider(response=_insight_response(summary="Revenue grew steadily."))

    result = interpret({"numeric_summary": {"columns": []}}, provider)

    assert result["is_valid"] is True
    assert result["insight"]["summary"] == "Revenue grew steadily."


def test_interpret_default_prompt_builder_parameter_is_build_prompt():
    signature = inspect.signature(interpret)

    assert signature.parameters["prompt_builder"].default is build_prompt


def test_interpret_without_prompt_builder_arg_uses_build_prompt(monkeypatch):
    provider = FakeProvider(response=_insight_response())
    payload = {"numeric_summary": {"columns": [{"name": "revenue", "sum": 100.0}]}}

    interpret(payload, provider)

    assert "financial data analyst assistant interpreting pre-calculated analytics results" in provider.last_prompt


# ==================================================
# D. Custom prompt builder
# ==================================================


def test_interpret_uses_supplied_comparison_prompt_builder():
    provider = FakeProvider(response=_insight_response())

    interpret(_ai_payload(), provider, prompt_builder=build_comparison_prompt)

    assert "comparison between two files/periods" in provider.last_prompt


def test_interpret_with_custom_prompt_builder_still_validates_malformed_json():
    provider = FakeProvider(response="not valid json")

    result = interpret(_ai_payload(), provider, prompt_builder=build_comparison_prompt)

    assert result["is_valid"] is False
    assert result["reason"] == "malformed_output"


def test_interpret_with_custom_prompt_builder_produces_valid_insight():
    provider = FakeProvider(response=_insight_response(
        summary="Budget_2024 increased compared with Budget_2025.",
        trend_interpretation="current is higher than previous",
    ))

    result = interpret(_ai_payload(), provider, prompt_builder=build_comparison_prompt)

    assert result["is_valid"] is True
    assert result["insight"]["trend_interpretation"] == "current is higher than previous"


# ==================================================
# E. Grounding
# ==================================================


def test_grounding_supplied_numeric_values_accepted():
    provider = FakeProvider(response=_insight_response(
        summary="The value rose from 100000.0 to 130000.0, a change of 30000.0.",
    ))

    result = interpret(_ai_payload(), provider, prompt_builder=build_comparison_prompt)

    assert result["is_valid"] is True


def test_grounding_negative_values_accepted():
    payload = _ai_payload(previous=100000.0, current=95000.0, absolute_change=-5000.0, percentage_change=-5.0)
    provider = FakeProvider(response=_insight_response(
        summary="The value fell from 100000.0 to 95000.0, a change of -5000.0 (-5.0%).",
    ))

    result = interpret(payload, provider, prompt_builder=build_comparison_prompt)

    assert result["is_valid"] is True


def test_grounding_decimal_values_accepted():
    payload = _ai_payload(previous=10.5, current=12.75, absolute_change=2.25, percentage_change=21.43)
    provider = FakeProvider(response=_insight_response(
        summary="The value increased from 10.5 to 12.75, a change of 2.25 (21.43%).",
    ))

    result = interpret(payload, provider, prompt_builder=build_comparison_prompt)

    assert result["is_valid"] is True


def test_grounding_zero_values_handled_correctly():
    payload = _ai_payload(previous=0.0, current=0.0, absolute_change=0.0, percentage_change=0.0)
    provider = FakeProvider(response=_insight_response(summary="Both files report a value of 0."))

    result = interpret(payload, provider, prompt_builder=build_comparison_prompt)

    assert result["is_valid"] is True


def test_grounding_percentage_values_accepted():
    provider = FakeProvider(response=_insight_response(summary="The value grew by 30.0%."))

    result = interpret(_ai_payload(), provider, prompt_builder=build_comparison_prompt)

    assert result["is_valid"] is True


def test_grounding_iso_periods_accepted():
    payload = _ai_payload(comparison_type="file_period_comparison", period="2024-03", metric=None)
    provider = FakeProvider(response=_insight_response(summary="The comparison covers period 2024-03."))

    result = interpret(payload, provider, prompt_builder=build_comparison_prompt)

    assert result["is_valid"] is True


def test_grounding_trend_label_comparison_accepted():
    payload = _ai_payload(
        comparison_type="file_trend_comparison", metric=None, column=None,
        previous="increasing", current="decreasing", absolute_change=None, percentage_change=None,
    )
    provider = FakeProvider(response=_insight_response(
        summary="Budget_2024's trend is increasing while Budget_2025's trend is decreasing.",
        trend_interpretation="the two files show opposite trend directions",
    ))

    result = interpret(payload, provider, prompt_builder=build_comparison_prompt)

    assert result["is_valid"] is True


def test_grounding_budget_2024_2025_display_names_echoed_accepted():
    provider = FakeProvider(response=_insight_response(
        summary="Budget_2024 increased compared with Budget_2025.",
    ))

    result = interpret(_ai_payload(), provider, prompt_builder=build_comparison_prompt)

    assert result["is_valid"] is True


def test_grounding_q1_2024_v2_label_echoed_accepted():
    payload = _ai_payload(
        file_a={"file_id": "aaa", "display_name": "Q1_2024_v2", "role": "previous"},
        file_b={"file_id": "bbb", "display_name": "Q2_2024_v2", "role": "current"},
    )
    provider = FakeProvider(response=_insight_response(
        summary="Q1_2024_v2 compares favorably with Q2_2024_v2.",
    ))

    result = interpret(payload, provider, prompt_builder=build_comparison_prompt)

    assert result["is_valid"] is True


def test_grounding_revenue2024_column_label_echoed_accepted():
    payload = _ai_payload(column="Revenue2024")
    provider = FakeProvider(response=_insight_response(
        summary="Revenue2024 increased from 100000.0 to 130000.0.",
    ))

    result = interpret(payload, provider, prompt_builder=build_comparison_prompt)

    assert result["is_valid"] is True


def test_grounding_unrelated_invented_number_rejected():
    provider = FakeProvider(response=_insight_response(summary="The value increased by 999999.99."))

    result = interpret(_ai_payload(), provider, prompt_builder=build_comparison_prompt)

    assert result["is_valid"] is False
    assert result["reason"] == "hallucinated_numeric_value"


def test_grounding_invented_percentage_rejected():
    provider = FakeProvider(response=_insight_response(summary="The value grew by an unprecedented 87.0%."))

    result = interpret(_ai_payload(), provider, prompt_builder=build_comparison_prompt)

    assert result["is_valid"] is False
    assert result["reason"] == "hallucinated_numeric_value"


def test_grounding_invented_decimal_rejected():
    provider = FakeProvider(response=_insight_response(summary="The value shifted by 12.34 units."))

    result = interpret(_ai_payload(), provider, prompt_builder=build_comparison_prompt)

    assert result["is_valid"] is False
    assert result["reason"] == "hallucinated_numeric_value"


def test_grounding_filename_digits_do_not_become_trusted_standalone_numbers():
    # CRITICAL: display_name "Budget_2024" is a known, maskable label, but
    # that must not create a blanket loophole for unrelated invented
    # numbers to slip through unpunished.
    payload = _ai_payload()  # file_a display_name is "Budget_2024"
    provider = FakeProvider(response=_insight_response(summary="The change was 9999 units."))

    result = interpret(payload, provider, prompt_builder=build_comparison_prompt)

    assert result["is_valid"] is False
    assert result["reason"] == "hallucinated_numeric_value"


def test_grounding_still_works_correctly_for_the_existing_single_file_payload_shape():
    # Regression: the label-masking addition must not change single-file
    # AI Insight behavior (build_prompt / default prompt_builder path).
    payload = {"numeric_summary": {"columns": [{"name": "revenue", "sum": 3266656.8}]}}
    provider = FakeProvider(response=_insight_response(
        summary="Revenue reached 3266656.8 in total.",
    ))

    result = interpret(payload, provider)

    assert result["is_valid"] is True


# ==================================================
# F. File ID safety
# ==================================================


def test_file_id_absent_from_llm_visible_comparison_payload():
    payload = _ai_payload()

    assert "f3a1c9e2b8d74a1f9c3e5b7a2d4f6081" not in json.dumps(payload)


def test_file_id_absent_from_provider_prompt():
    provider = FakeProvider(response=_insight_response())

    interpret(_ai_payload(), provider, prompt_builder=build_comparison_prompt)

    assert "f3a1c9e2b8d74a1f9c3e5b7a2d4f6081" not in provider.last_prompt
    assert "9bd27a41e6c34f2ab1d0987654321abc" not in provider.last_prompt


# ==================================================
# G. Failure handling
# ==================================================


def _maybe_interpret_comparison(comparison_result: dict, provider) -> dict | None:
    """Documents/proves the intended call-site rule: AI interpretation is
    only attempted for a successful comparison. This mirrors exactly how a
    future comparison call site (e.g. app.py) must gate the call — it is
    test-only, not new production code."""
    if not comparison_result["found"]:
        return None
    ai_payload = build_comparison_ai_payload(comparison_result)
    return interpret(ai_payload, provider, prompt_builder=build_comparison_prompt)


def test_found_false_comparison_does_not_call_provider_at_the_call_site():
    provider = FakeProvider(response=_insight_response())
    failure_result = {
        "found": False, "reason": "file_not_found", "comparison_type": "file_value_comparison",
        "metric": None, "column": None, "period": None, "file_a": None, "file_b": None,
        "previous": None, "current": None, "absolute_change": None, "percentage_change": None,
        "is_valid": None, "extra": {},
    }

    result = _maybe_interpret_comparison(failure_result, provider)

    assert result is None
    assert provider.call_count == 0


def test_direct_interpret_call_with_failure_shaped_payload_does_not_crash():
    failure_result = {
        "found": False, "reason": "file_not_found", "comparison_type": "file_value_comparison",
        "metric": None, "column": None, "period": None, "file_a": None, "file_b": None,
        "previous": None, "current": None, "absolute_change": None, "percentage_change": None,
        "is_valid": None, "extra": {},
    }
    ai_payload = build_comparison_ai_payload(failure_result)
    provider = FakeProvider(response=_insight_response(summary="No comparison could be made."))

    result = interpret(ai_payload, provider, prompt_builder=build_comparison_prompt)

    assert result["is_valid"] is True  # no crash; nothing in the payload to hallucinate against


# ==================================================
# H. Provider failures
# ==================================================


def test_provider_error_behavior_unchanged_with_comparison_prompt_builder():
    provider = FakeProvider(error=AIProviderError("missing_api_key"))

    result = interpret(_ai_payload(), provider, prompt_builder=build_comparison_prompt)

    assert result["is_valid"] is False
    assert result["reason"] == "missing_api_key"
    assert result["insight"] is None


# ==================================================
# I. Malformed output
# ==================================================


def test_malformed_json_still_returns_malformed_output_with_comparison_prompt_builder():
    provider = FakeProvider(response="this is not json")

    result = interpret(_ai_payload(), provider, prompt_builder=build_comparison_prompt)

    assert result["is_valid"] is False
    assert result["reason"] == "malformed_output"


def test_invalid_output_shape_still_returns_malformed_output_with_comparison_prompt_builder():
    provider = FakeProvider(response=json.dumps({"summary": "ok", "key_insights": []}))  # missing keys

    result = interpret(_ai_payload(), provider, prompt_builder=build_comparison_prompt)

    assert result["is_valid"] is False
    assert result["reason"] == "malformed_output"


# ==================================================
# J. No mutation (end-to-end through the whole chain)
# ==================================================


def test_comparison_result_unchanged_across_the_full_chain():
    comparison_result = _comparison_result()
    before = json.loads(json.dumps(comparison_result))  # deep, order-independent snapshot

    ai_payload = build_comparison_ai_payload(comparison_result)
    build_comparison_prompt(ai_payload)
    provider = FakeProvider(response=_insight_response())
    interpret(ai_payload, provider, prompt_builder=build_comparison_prompt)

    assert json.loads(json.dumps(comparison_result)) == before


# ==================================================
# K. Provider call count
# ==================================================


def test_provider_called_exactly_once_for_successful_comparison_interpretation():
    provider = FakeProvider(response=_insight_response())

    interpret(_ai_payload(), provider, prompt_builder=build_comparison_prompt)

    assert provider.call_count == 1
