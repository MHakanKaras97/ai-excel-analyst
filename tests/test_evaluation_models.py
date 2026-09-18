from src.evaluation.models import EvalCase, EvalResult


def test_eval_case_round_trips_through_dict():
    case = EvalCase(case_id="c1", category="column_resolution", input={"a": 1}, expected={"b": 2})

    data = case.to_dict()
    restored = EvalCase.from_dict(data)

    assert restored == case


def test_eval_case_from_dict_defaults_missing_optional_fields():
    restored = EvalCase.from_dict({"case_id": "c1", "category": "column_resolution"})

    assert restored.input == {}
    assert restored.expected == {}
    assert restored.metadata == {}


def test_eval_result_to_dict_shape():
    result = EvalResult(case_id="c1", category="column_resolution", passed=True, actual={"a": 1}, expected={"a": 1})

    assert result.to_dict() == {
        "case_id": "c1", "category": "column_resolution", "passed": True,
        "actual": {"a": 1}, "expected": {"a": 1}, "reason": None,
    }


def test_eval_result_carries_reason_when_failed():
    result = EvalResult(case_id="c1", category="column_resolution", passed=False,
                         actual={"a": 1}, expected={"a": 2}, reason="mismatch")

    assert result.to_dict()["reason"] == "mismatch"
