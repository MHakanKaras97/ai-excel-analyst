"""Lightweight evaluation runner (V0.8.3).

Loads EvalCase records from a benchmark JSON file and runs each one
through the deterministic component its `category` names — never a live
LLM call: any case that needs a "model response" supplies it as fixed
input text (`llm_response`), exercising the exact same closed-schema
validation / deterministic-resolution / grounding code paths a real
response would go through, without any network dependency or
non-determinism. This is a benchmark of the deterministic and validation
layers, not of any particular model's judgment.
"""
import json
from pathlib import Path

from src.ai_interpreter import interpret as interpret_insight
from src.ai_provider import AIProviderError
from src.chart_intent_interpreter import interpret_chart_question
from src.evaluation.models import EvalCase, EvalResult
from src.evidence.validator import contains_unsupported_causal_claim, validate_referenced_evidence_ids
from src.qa_engine import resolve_column, resolve_period
from src.qa_interpreter import interpret_question
from src.schema.schema_resolver import resolve_semantic_column


class _FixedResponseProvider:
    """A minimal AIProvider double returning one fixed response (or raising
    one fixed AIProviderError) — the evaluation-harness equivalent of the
    FakeProvider test doubles used throughout the test suite."""

    def __init__(self, response: str | None = None, error_reason: str | None = None):
        self._response = response
        self._error_reason = error_reason

    def generate(self, prompt: str) -> str:
        if self._error_reason is not None:
            raise AIProviderError(self._error_reason)
        return self._response


def load_cases(path) -> list[EvalCase]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [EvalCase.from_dict(item) for item in data]


def _matches_expected(actual: dict, expected: dict) -> bool:
    """`expected` only needs to name the keys the case cares about — actual
    may legitimately carry extra fields (e.g. "candidates") the benchmark
    doesn't assert on."""
    return all(actual.get(key) == value for key, value in expected.items())


def _provider_from_input(input_data: dict) -> _FixedResponseProvider:
    if input_data.get("provider_error"):
        return _FixedResponseProvider(error_reason=input_data["provider_error"])
    return _FixedResponseProvider(response=input_data.get("llm_response"))


def _evaluate_column_resolution(case: EvalCase) -> dict:
    return resolve_column(case.input.get("column_hint"), case.input.get("column_names") or [])


def _evaluate_period_resolution(case: EvalCase) -> dict:
    return resolve_period(case.input.get("period_hint"), case.input.get("period_labels") or [])


def _evaluate_semantic_resolution(case: EvalCase) -> dict:
    return resolve_semantic_column(case.input.get("term"), case.input.get("schema_columns") or [])


def _evaluate_intent_classification(case: EvalCase) -> dict:
    provider = _provider_from_input(case.input)
    return interpret_question(case.input["question"], case.input.get("column_names") or [], provider)


def _evaluate_chart_intent_classification(case: EvalCase) -> dict:
    provider = _provider_from_input(case.input)
    return interpret_chart_question(case.input["question"], case.input.get("column_names") or [], provider)


def _evaluate_grounding(case: EvalCase) -> dict:
    provider = _provider_from_input(case.input)
    return interpret_insight(case.input.get("analytics_payload") or {}, provider)


def _evaluate_safety(case: EvalCase) -> dict:
    check = case.input.get("check")
    if check == "causal_claim":
        return {"flagged": contains_unsupported_causal_claim(case.input.get("text", ""))}
    if check == "evidence_reference":
        result = validate_referenced_evidence_ids(
            case.input.get("referenced_ids") or [], case.input.get("evidence_list") or [],
        )
        return {"valid": result["valid"]}
    if check in ("unsupported_question", "malformed_response", "provider_failure"):
        return _evaluate_intent_classification(case)
    return {"reason": "unknown_check"}


_EVALUATORS = {
    "column_resolution": _evaluate_column_resolution,
    "period_resolution": _evaluate_period_resolution,
    "semantic_resolution": _evaluate_semantic_resolution,
    "intent_classification": _evaluate_intent_classification,
    "chart_intent_classification": _evaluate_chart_intent_classification,
    "grounding": _evaluate_grounding,
    "safety": _evaluate_safety,
}


def run_case(case: EvalCase) -> EvalResult:
    evaluator = _EVALUATORS.get(case.category)
    if evaluator is None:
        return EvalResult(case.case_id, case.category, False, {}, case.expected, reason="unknown_category")

    try:
        actual = evaluator(case)
    except Exception as exc:
        # A malformed benchmark case (or a genuine regression) must show up
        # as a reported failure, never crash the whole evaluation run.
        return EvalResult(
            case.case_id, case.category, False, {"exception": repr(exc)}, case.expected, reason="evaluator_error",
        )

    passed = _matches_expected(actual, case.expected)
    return EvalResult(case.case_id, case.category, passed, actual, case.expected, reason=None if passed else "mismatch")


def run_cases(cases: list[EvalCase]) -> list[EvalResult]:
    return [run_case(case) for case in cases]


def run_benchmark_file(path) -> list[EvalResult]:
    return run_cases(load_cases(path))
