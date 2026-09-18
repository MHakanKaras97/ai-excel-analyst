"""Aggregate metrics over a list of EvalResult (V0.8.3)."""
from src.evaluation.models import EvalResult


def summarize(results: list[EvalResult]) -> dict:
    total = len(results)
    passed = sum(1 for r in results if r.passed)

    by_category: dict[str, dict] = {}
    for result in results:
        bucket = by_category.setdefault(result.category, {"total": 0, "passed": 0})
        bucket["total"] += 1
        if result.passed:
            bucket["passed"] += 1

    for bucket in by_category.values():
        bucket["accuracy"] = bucket["passed"] / bucket["total"] if bucket["total"] else None

    return {
        "total": total,
        "passed": passed,
        "accuracy": (passed / total) if total else None,
        "by_category": by_category,
    }
