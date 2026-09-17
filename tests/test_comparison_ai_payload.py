import copy
import json

from src.comparison_ai_payload import build_comparison_ai_payload


def _success_comparison_result(**overrides):
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


def _failure_comparison_result(**overrides):
    result = {
        "found": False, "reason": "file_not_found", "comparison_type": "file_value_comparison",
        "metric": None, "column": None, "period": None,
        "file_a": None, "file_b": None,
        "previous": None, "current": None, "absolute_change": None,
        "percentage_change": None, "is_valid": None, "extra": {},
    }
    result.update(overrides)
    return result


def test_build_comparison_ai_payload_removes_file_id():
    payload = build_comparison_ai_payload(_success_comparison_result())

    assert "file_id" not in payload["file_a"]
    assert "file_id" not in payload["file_b"]


def test_build_comparison_ai_payload_preserves_display_name():
    payload = build_comparison_ai_payload(_success_comparison_result())

    assert payload["file_a"]["display_name"] == "Budget_2024"
    assert payload["file_b"]["display_name"] == "Budget_2025"


def test_build_comparison_ai_payload_preserves_role():
    payload = build_comparison_ai_payload(_success_comparison_result())

    assert payload["file_a"]["role"] == "previous"
    assert payload["file_b"]["role"] == "current"


def test_build_comparison_ai_payload_shape_is_exactly_display_name_and_role():
    payload = build_comparison_ai_payload(_success_comparison_result())

    assert set(payload["file_a"].keys()) == {"display_name", "role"}
    assert set(payload["file_b"].keys()) == {"display_name", "role"}


def test_build_comparison_ai_payload_does_not_mutate_original_result():
    comparison_result = _success_comparison_result()
    before = copy.deepcopy(comparison_result)

    build_comparison_ai_payload(comparison_result)

    assert comparison_result == before
    assert "file_id" in comparison_result["file_a"]  # original untouched


def test_build_comparison_ai_payload_handles_found_true():
    payload = build_comparison_ai_payload(_success_comparison_result())

    assert payload["found"] is True
    assert payload["previous"] == 100000.0
    assert payload["current"] == 130000.0
    assert payload["absolute_change"] == 30000.0
    assert payload["percentage_change"] == 30.0


def test_build_comparison_ai_payload_handles_found_false_without_crashing():
    payload = build_comparison_ai_payload(_failure_comparison_result())

    assert payload["found"] is False
    assert payload["reason"] == "file_not_found"
    assert payload["file_a"] is None
    assert payload["file_b"] is None


def test_build_comparison_ai_payload_preserves_required_comparison_facts():
    comparison_result = _success_comparison_result(
        comparison_type="file_period_comparison", period="2024-03", metric=None,
    )

    payload = build_comparison_ai_payload(comparison_result)

    assert payload["comparison_type"] == "file_period_comparison"
    assert payload["period"] == "2024-03"
    assert payload["column"] == "TotalPrice"
    assert payload["is_valid"] is True
    assert payload["extra"] == {}


def test_build_comparison_ai_payload_preserves_extra_detail():
    comparison_result = _success_comparison_result(
        comparison_type="file_trend_comparison",
        previous="increasing", current="decreasing", absolute_change=None, percentage_change=None,
        extra={"file_a_trend": {"trend": "increasing"}, "file_b_trend": {"trend": "decreasing"}},
    )

    payload = build_comparison_ai_payload(comparison_result)

    assert payload["extra"]["file_a_trend"]["trend"] == "increasing"
    assert payload["extra"]["file_b_trend"]["trend"] == "decreasing"


def test_build_comparison_ai_payload_produces_json_serializable_output():
    json.dumps(build_comparison_ai_payload(_success_comparison_result()))
    json.dumps(build_comparison_ai_payload(_failure_comparison_result()))


def test_build_comparison_ai_payload_does_not_add_derived_numeric_values():
    comparison_result = _success_comparison_result()
    payload = build_comparison_ai_payload(comparison_result)

    assert set(payload.keys()) == set(comparison_result.keys())
