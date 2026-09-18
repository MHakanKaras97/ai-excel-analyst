"""End-to-end Q&A pipeline test (V0.5.5).

Exercises the full chain exactly as app.py will drive it:

    question -> qa_interpreter.interpret_question() -> structured intent
             -> qa_engine.dispatch_intent()          -> grounded result
             -> qa_answer.answer_grounded_result()   -> final answer

A FakeProvider stands in for the LLM boundary — no live Gemini call is made.
"""
import copy
import json

import pandas as pd

from src.ai_provider import AIProviderError
from src.qa_answer import answer_grounded_result
from src.qa_engine import dispatch_intent
from src.qa_interpreter import interpret_question

DECOY_NUMBER = "999999.99"


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


def _numeric_summary():
    return {
        "columns": [
            {"name": "TotalPrice", "count": 3, "missing_count": 0, "sum": 450.0,
             "mean": 150.0, "median": 150.0, "min": 100.0, "max": 200.0, "std": 50.0},
        ],
    }


def _monthly_series():
    return pd.Series([100.0, 200.0, 150.0], index=["2024-01", "2024-02", "2024-03"], name="TotalPrice")


def _analysis_payload():
    return {"numeric_summary": _numeric_summary()}


def _run_pipeline(question, column_names, provider, analysis_payload, monthly_series):
    """Mirrors exactly the sequence app.py's Q&A section runs."""
    interpretation = interpret_question(question, column_names, provider)
    if not interpretation["is_valid"]:
        return answer_grounded_result(interpretation)

    grounded_result = dispatch_intent(interpretation["intent"], analysis_payload, monthly_series)
    return answer_grounded_result(grounded_result)


# --- Happy path -----------------------------------------------------------


def test_pipeline_end_to_end_success():
    intent_response = json.dumps({
        "intent": "period_value", "metric": None, "column_hint": "TotalPrice",
        "period_hint": "2024-03", "from_period_hint": None, "to_period_hint": None,
    })
    provider = FakeProvider(response=intent_response)

    answer = _run_pipeline(
        "What was TotalPrice in March 2024?",
        ["TotalPrice"],
        provider,
        _analysis_payload(),
        _monthly_series(),
    )

    assert provider.call_count == 1
    assert answer == "TotalPrice in 2024-03 was 150.0."


def test_pipeline_no_hallucinated_numeric_value_can_enter_the_final_answer():
    # The LLM's JSON response smuggles a plausible-looking decoy number
    # inside a free-text hint field. Since hints are only ever used to
    # resolve against known columns/periods (never emitted verbatim as a
    # numeric answer), the decoy must never appear in the final answer.
    intent_response = json.dumps({
        "intent": "period_value", "metric": None,
        "column_hint": f"TotalPrice (allegedly {DECOY_NUMBER})",
        "period_hint": "2024-03", "from_period_hint": None, "to_period_hint": None,
    })
    provider = FakeProvider(response=intent_response)

    answer = _run_pipeline(
        "What was TotalPrice in March 2024?",
        ["TotalPrice"],
        provider,
        _analysis_payload(),
        _monthly_series(),
    )

    assert DECOY_NUMBER not in answer
    # The decoy column_hint doesn't case-insensitively exact-match "TotalPrice"
    # and isn't a close-enough fuzzy match, so resolution correctly fails
    # rather than silently answering against the wrong (or right) column.
    assert answer == "I couldn't find a column matching that question."


def test_pipeline_period_extremum_end_to_end():
    intent_response = json.dumps({
        "intent": "period_extremum", "metric": "max", "column_hint": None,
        "period_hint": None, "from_period_hint": None, "to_period_hint": None,
    })
    provider = FakeProvider(response=intent_response)

    answer = _run_pipeline(
        "Which month had the highest TotalPrice?",
        ["TotalPrice"],
        provider,
        _analysis_payload(),
        _monthly_series(),
    )

    assert provider.call_count == 1
    assert "highest" in answer
    assert "200.0" in answer
    assert "2024-02" in answer


# --- Failure paths ----------------------------------------------------


def test_pipeline_provider_error_produces_deterministic_answer():
    provider = FakeProvider(error=AIProviderError("provider_error"))

    answer = _run_pipeline(
        "What was TotalPrice in March?",
        ["TotalPrice"],
        provider,
        _analysis_payload(),
        _monthly_series(),
    )

    assert provider.call_count == 1
    assert answer == "The question couldn't be interpreted because the AI provider returned an error."


def test_pipeline_malformed_llm_response_produces_deterministic_answer():
    provider = FakeProvider(response="not valid json at all")

    answer = _run_pipeline(
        "What was TotalPrice in March?",
        ["TotalPrice"],
        provider,
        _analysis_payload(),
        _monthly_series(),
    )

    assert provider.call_count == 1
    assert answer == "The question couldn't be interpreted (the AI response wasn't valid JSON)."


def test_pipeline_unsupported_question_produces_deterministic_answer():
    intent_response = json.dumps({
        "intent": "unsupported", "metric": None, "column_hint": None,
        "period_hint": None, "from_period_hint": None, "to_period_hint": None,
    })
    provider = FakeProvider(response=intent_response)

    answer = _run_pipeline(
        "What's the weather like today?",
        ["TotalPrice"],
        provider,
        _analysis_payload(),
        _monthly_series(),
    )

    assert provider.call_count == 1
    assert answer == "That question isn't supported yet."


def test_pipeline_engine_failure_produces_deterministic_answer_without_second_llm_call():
    # period_hint doesn't resolve — the LLM is never consulted again to
    # "fix" or guess the period; exactly one call total for the question.
    intent_response = json.dumps({
        "intent": "period_value", "metric": None, "column_hint": "TotalPrice",
        "period_hint": "December", "from_period_hint": None, "to_period_hint": None,
    })
    provider = FakeProvider(response=intent_response)

    answer = _run_pipeline(
        "What was TotalPrice in December?",
        ["TotalPrice"],
        provider,
        _analysis_payload(),
        _monthly_series(),
    )

    assert provider.call_count == 1
    assert answer == "I couldn't find a period matching that question."


# --- Reported bug regression: column_stat restricted to a calendar year --


def _quantity_monthly_series():
    # Spans 2023-2025 so a "2024" restriction has real other-year data to
    # incorrectly include if the year filter were ever lost again.
    return pd.Series(
        [500.0, 600.0, 700.0, 800.0, 900.0, 6260.0, 800.0, 400.0, 560.0],
        index=["2023-06", "2023-12", "2024-01", "2024-02", "2024-03",
               "2024-04", "2024-05", "2025-01", "2025-06"],
        name="Quantity",
    )


def _quantity_analysis_payload():
    return {"numeric_summary": {"columns": [
        {"name": "Quantity", "count": 9, "missing_count": 0, "sum": 11520.0,
         "mean": 1280.0, "median": 700.0, "min": 400.0, "max": 6260.0, "std": 1856.0},
    ]}}


def test_pipeline_reported_bug_total_quantity_in_2024_is_year_restricted():
    # Exact reported request: "What was the total quantity purchased in
    # 2024?" previously answered with the whole-dataset sum (8260.0-style
    # figure) instead of only 2024's rows.
    intent_response = json.dumps({
        "intent": "column_stat", "metric": "sum", "column_hint": "Quantity",
        "period_hint": "2024", "from_period_hint": None, "to_period_hint": None,
    })
    provider = FakeProvider(response=intent_response)

    answer = _run_pipeline(
        "What was the total quantity purchased in 2024?",
        ["Quantity"],
        provider,
        _quantity_analysis_payload(),
        _quantity_monthly_series(),
    )

    assert provider.call_count == 1
    # 700 + 800 + 900 + 6260 + 800 (2024-01..2024-05 only)
    assert answer == "The sum of Quantity is 9460.0."
    # The full-dataset sum (11520.0) must never appear in a year-restricted answer.
    assert "11520" not in answer


# --- Cross-cutting guarantees -----------------------------------------


def test_pipeline_does_not_mutate_inputs():
    intent_response = json.dumps({
        "intent": "period_value", "metric": None, "column_hint": "TotalPrice",
        "period_hint": "2024-03", "from_period_hint": None, "to_period_hint": None,
    })
    provider = FakeProvider(response=intent_response)
    payload = _analysis_payload()
    series = _monthly_series()
    payload_before = copy.deepcopy(payload)
    series_before = series.copy()
    column_names = ["TotalPrice"]
    column_names_before = list(column_names)

    _run_pipeline("What was TotalPrice in March 2024?", column_names, provider, payload, series)

    assert payload == payload_before
    assert series.equals(series_before)
    assert column_names == column_names_before
