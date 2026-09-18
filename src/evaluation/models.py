"""Evaluation data model (V0.8.1).

Plain, JSON-serializable contracts for benchmark cases and results. No
framework, no external dependency beyond the stdlib — a benchmark case is
just structured data describing an input and the deterministically
verifiable outcome expected from it.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalCase:
    """One benchmark case.

    `category` selects which evaluator the runner dispatches to (e.g.
    "intent_classification", "column_resolution", "grounding"). `input` and
    `expected` are free-form dicts whose shape is defined by that category
    — kept as plain dicts rather than per-category subclasses so new
    categories don't require new model classes.
    """
    case_id: str
    category: str
    input: dict
    expected: dict
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "input": self.input,
            "expected": self.expected,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(data: dict) -> "EvalCase":
        return EvalCase(
            case_id=data["case_id"],
            category=data["category"],
            input=data.get("input") or {},
            expected=data.get("expected") or {},
            metadata=data.get("metadata") or {},
        )


@dataclass(frozen=True)
class EvalResult:
    """The outcome of running one EvalCase through its evaluator."""
    case_id: str
    category: str
    passed: bool
    actual: dict
    expected: dict
    reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "passed": self.passed,
            "actual": self.actual,
            "expected": self.expected,
            "reason": self.reason,
        }
