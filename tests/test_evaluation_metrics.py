from pathlib import Path

from src.evaluation.metrics import summarize
from src.evaluation.models import EvalResult
from src.evaluation.reports import format_failures, format_summary
from src.evaluation.runner import load_cases, run_case, run_cases

BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "data" / "evaluation"


def _results():
    return [
        EvalResult("c1", "column_resolution", True, {}, {}),
        EvalResult("c2", "column_resolution", False, {}, {}, reason="mismatch"),
        EvalResult("c3", "period_resolution", True, {}, {}),
    ]


def test_summarize_overall_and_per_category():
    summary = summarize(_results())

    assert summary["total"] == 3
    assert summary["passed"] == 2
    assert summary["accuracy"] == 2 / 3
    assert summary["by_category"]["column_resolution"] == {"total": 2, "passed": 1, "accuracy": 0.5}
    assert summary["by_category"]["period_resolution"] == {"total": 1, "passed": 1, "accuracy": 1.0}


def test_summarize_empty_results():
    summary = summarize([])

    assert summary["total"] == 0
    assert summary["accuracy"] is None
    assert summary["by_category"] == {}


def test_format_summary_includes_category_lines():
    summary = summarize(_results())

    text = format_summary(summary)

    assert "2/3" in text
    assert "column_resolution" in text
    assert "period_resolution" in text


def test_format_summary_handles_no_cases():
    assert format_summary(summarize([])) == "Overall: no cases run"


def test_format_failures_lists_only_failed_cases():
    text = format_failures(_results())

    assert "c2" in text
    assert "c1" not in text
    assert "c3" not in text


def test_format_failures_handles_no_failures():
    results = [EvalResult("c1", "column_resolution", True, {}, {})]

    assert format_failures(results) == "No failed cases."


# ==================================================
# Runner against real benchmark files (the actual V0.8.2 dataset)
# ==================================================


def test_qa_cases_benchmark_passes_fully():
    cases = load_cases(BENCHMARK_DIR / "qa_cases.json")
    results = run_cases(cases)

    failures = [r for r in results if not r.passed]
    assert failures == [], format_failures(results)
    assert len(cases) == 20


def test_chart_cases_benchmark_passes_fully():
    results = run_cases(load_cases(BENCHMARK_DIR / "chart_cases.json"))

    failures = [r for r in results if not r.passed]
    assert failures == [], format_failures(results)


def test_semantic_cases_benchmark_passes_fully():
    results = run_cases(load_cases(BENCHMARK_DIR / "semantic_cases.json"))

    failures = [r for r in results if not r.passed]
    assert failures == [], format_failures(results)
    assert len(results) == 20


def test_grounding_cases_benchmark_passes_fully():
    results = run_cases(load_cases(BENCHMARK_DIR / "grounding_cases.json"))

    failures = [r for r in results if not r.passed]
    assert failures == [], format_failures(results)
    assert len(results) == 20


def test_safety_cases_benchmark_passes_fully():
    results = run_cases(load_cases(BENCHMARK_DIR / "safety_cases.json"))

    failures = [r for r in results if not r.passed]
    assert failures == [], format_failures(results)
    assert len(results) == 15


def test_unknown_category_is_reported_not_crashed():
    from src.evaluation.models import EvalCase

    case = EvalCase(case_id="x", category="not_a_real_category", input={}, expected={})

    result = run_case(case)

    assert result.passed is False
    assert result.reason == "unknown_category"


def test_evaluator_exception_is_reported_not_raised():
    from src.evaluation.models import EvalCase

    # Missing the required "question" key triggers a KeyError inside the
    # evaluator — the runner must catch it and report a failure, not crash.
    case = EvalCase(case_id="x", category="intent_classification", input={}, expected={"is_valid": True})

    result = run_case(case)

    assert result.passed is False
    assert result.reason == "evaluator_error"
