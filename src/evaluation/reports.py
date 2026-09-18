"""Plain-text rendering of evaluation results (V0.8.3). No dashboards, no
experiment-tracking integration — just a readable summary for a report or
the terminal.
"""
from src.evaluation.models import EvalResult


def format_summary(summary: dict) -> str:
    if summary["total"] == 0:
        return "Overall: no cases run"

    lines = [f"Overall: {summary['passed']}/{summary['total']} ({summary['accuracy']:.1%})"]
    for category in sorted(summary["by_category"]):
        bucket = summary["by_category"][category]
        lines.append(f"  {category}: {bucket['passed']}/{bucket['total']} ({bucket['accuracy']:.1%})")
    return "\n".join(lines)


def format_failures(results: list[EvalResult]) -> str:
    failures = [r for r in results if not r.passed]
    if not failures:
        return "No failed cases."

    lines = []
    for result in failures:
        lines.append(f"{result.case_id} ({result.category}): expected {result.expected}, got {result.actual} [{result.reason}]")
    return "\n".join(lines)
