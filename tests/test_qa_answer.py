import copy
import json

from src.qa_answer import answer_grounded_result
from src.qa_engine import dispatch_intent

# ==================================================
# SUCCESSFUL INTENTS
# ==================================================


def test_answer_period_value_success():
    result = {
        "found": True, "reason": None, "intent": "period_value",
        "column": "TotalPrice", "period": "2024-03", "value": 12345.67, "extra": {},
    }

    answer = answer_grounded_result(result)

    assert "TotalPrice" in answer
    assert "2024-03" in answer
    assert "12345.67" in answer


def test_answer_period_extremum_min_says_lowest():
    result = {
        "found": True, "reason": None, "intent": "period_extremum",
        "column": "TotalPrice", "period": "2024-01", "value": 100.0, "extra": {"metric": "min"},
    }

    answer = answer_grounded_result(result)

    assert "lowest" in answer
    assert "TotalPrice" in answer
    assert "2024-01" in answer
    assert "100.0" in answer


def test_answer_period_extremum_max_says_highest():
    result = {
        "found": True, "reason": None, "intent": "period_extremum",
        "column": "TotalPrice", "period": "2024-02", "value": 200.0, "extra": {"metric": "max"},
    }

    answer = answer_grounded_result(result)

    assert "highest" in answer
    assert "200.0" in answer


def test_answer_period_change_success():
    result = {
        "found": True, "reason": None, "intent": "period_change",
        "column": "TotalPrice", "period": None, "value": 100.0,
        "extra": {
            "from_period": "2024-01", "to_period": "2024-02",
            "previous": 100.0, "current": 200.0,
            "absolute_change": 100.0, "percentage_change": 100.0,
            "is_valid": True,
        },
    }

    answer = answer_grounded_result(result)

    assert "2024-01" in answer
    assert "2024-02" in answer
    assert "100.0" in answer
    assert "200.0" in answer
    assert "100.0%" in answer or "100.0" in answer


def test_answer_period_change_without_percentage_change_still_answers():
    result = {
        "found": True, "reason": None, "intent": "period_change",
        "column": "TotalPrice", "period": None, "value": 0.0,
        "extra": {
            "from_period": "2024-01", "to_period": "2024-02",
            "previous": 0.0, "current": 0.0,
            "absolute_change": 0.0, "percentage_change": None,
            "is_valid": True,
        },
    }

    answer = answer_grounded_result(result)

    assert isinstance(answer, str)
    assert "%" not in answer


def test_answer_column_stat_success():
    result = {
        "found": True, "reason": None, "intent": "column_stat",
        "column": "TotalPrice", "period": None, "value": 450.0, "extra": {"metric": "sum"},
    }

    answer = answer_grounded_result(result)

    assert "sum" in answer
    assert "TotalPrice" in answer
    assert "450.0" in answer


def test_answer_anomaly_check_with_anomalies():
    result = {
        "found": True, "reason": None, "intent": "anomaly_check",
        "column": "TotalPrice", "period": None, "value": 1,
        "extra": {"insufficient_data": False, "lower_bound": 10.0, "upper_bound": 190.0,
                  "anomalies": [{"position": 1, "period": "2024-02", "value": 200.0, "direction": "high"}]},
    }

    answer = answer_grounded_result(result)

    assert "1 anomaly" in answer
    assert "TotalPrice" in answer


def test_answer_anomaly_check_with_no_anomalies():
    result = {
        "found": True, "reason": None, "intent": "anomaly_check",
        "column": "TotalPrice", "period": None, "value": 0,
        "extra": {"insufficient_data": False, "lower_bound": 10.0, "upper_bound": 300.0, "anomalies": []},
    }

    answer = answer_grounded_result(result)

    assert "No anomalies" in answer


def test_answer_anomaly_check_insufficient_data():
    result = {
        "found": True, "reason": None, "intent": "anomaly_check",
        "column": "TotalPrice", "period": None, "value": 0,
        "extra": {"insufficient_data": True, "lower_bound": None, "upper_bound": None, "anomalies": []},
    }

    answer = answer_grounded_result(result)

    assert "not enough data" in answer


def test_answer_missing_values_with_missing_columns():
    result = {
        "found": True, "reason": None, "intent": "missing_values",
        "column": None, "period": None, "value": 1,
        "extra": {"columns": [{"name": "Region", "missing_count": 2, "missing_percentage": 20.0}]},
    }

    answer = answer_grounded_result(result)

    assert "Region" in answer
    assert "2" in answer


def test_answer_missing_values_with_no_missing_columns():
    result = {
        "found": True, "reason": None, "intent": "missing_values",
        "column": None, "period": None, "value": 0, "extra": {"columns": []},
    }

    answer = answer_grounded_result(result)

    assert "No columns have missing values" in answer


# ==================================================
# FAILURE REASONS
# ==================================================

GROUNDED_FAILURE_REASONS = [
    "column_not_found",
    "period_not_found",
    "ambiguous_column",
    "ambiguous_period",
    "trend_not_computed",
    "profile_not_computed",
    "column_series_mismatch",
    "unsupported",
]

INTERPRETER_FAILURE_REASONS = [
    "invalid_json",
    "invalid_schema",
    "provider_error",
]


def test_answer_covers_every_grounded_failure_reason():
    for reason in GROUNDED_FAILURE_REASONS:
        result = {
            "found": False, "reason": reason, "intent": "period_value",
            "column": None, "period": None, "value": None, "extra": {},
        }

        answer = answer_grounded_result(result)

        assert isinstance(answer, str)
        assert len(answer) > 0
        assert answer != "That question couldn't be answered."  # each reason has its own message


def test_answer_covers_every_interpreter_failure_reason():
    for reason in INTERPRETER_FAILURE_REASONS:
        result = {"is_valid": False, "reason": reason}

        answer = answer_grounded_result(result)

        assert isinstance(answer, str)
        assert len(answer) > 0
        assert answer != "That question couldn't be answered."


def test_answer_unknown_reason_falls_back_to_default_message():
    result = {"found": False, "reason": "some_future_reason", "intent": "period_value",
              "column": None, "period": None, "value": None, "extra": {}}

    assert answer_grounded_result(result) == "That question couldn't be answered."


def test_answer_ambiguous_column_includes_candidates():
    result = {
        "found": False, "reason": "ambiguous_column", "intent": "period_value",
        "column": None, "period": None, "value": None,
        "extra": {"candidates": ["Revenue", "Revenue_USD"]},
    }

    answer = answer_grounded_result(result)

    assert "Revenue" in answer
    assert "Revenue_USD" in answer


def test_answer_ambiguous_period_includes_candidates():
    result = {
        "found": False, "reason": "ambiguous_period", "intent": "period_value",
        "column": None, "period": None, "value": None,
        "extra": {"candidates": ["2023-03", "2024-03"]},
    }

    answer = answer_grounded_result(result)

    assert "2023-03" in answer
    assert "2024-03" in answer


def test_answer_interpreter_success_shape_is_not_treated_as_terminal():
    # {"is_valid": True, "intent": {...}} is an interpreter SUCCESS, not a
    # terminal answer point — it should never reach answer_grounded_result
    # in the real pipeline, but must not crash if it somehow does.
    result = {"is_valid": True, "intent": {"intent": "period_value"}}

    answer = answer_grounded_result(result)

    assert isinstance(answer, str)


def test_answer_handles_non_dict_input_gracefully():
    assert answer_grounded_result(None) == "That question couldn't be answered."
    assert answer_grounded_result("not a dict") == "That question couldn't be answered."


# ==================================================
# CROSS-CUTTING GUARANTEES
# ==================================================


def test_answer_preserves_exact_numeric_value():
    result = {
        "found": True, "reason": None, "intent": "column_stat",
        "column": "TotalPrice", "period": None, "value": 3266656.789123, "extra": {"metric": "sum"},
    }

    answer = answer_grounded_result(result)

    assert "3266656.789123" in answer


def test_answer_does_not_mutate_input():
    result = {
        "found": True, "reason": None, "intent": "period_value",
        "column": "TotalPrice", "period": "2024-03", "value": 150.0, "extra": {},
    }
    before = copy.deepcopy(result)

    answer_grounded_result(result)

    assert result == before


def test_answer_is_deterministic_across_repeated_calls():
    result = {
        "found": True, "reason": None, "intent": "anomaly_check",
        "column": "TotalPrice", "period": None, "value": 1,
        "extra": {"insufficient_data": False, "lower_bound": 10.0, "upper_bound": 190.0,
                  "anomalies": [{"position": 1, "period": "2024-02", "value": 200.0, "direction": "high"}]},
    }

    first = answer_grounded_result(result)
    second = answer_grounded_result(result)

    assert first == second


def test_answer_accepts_actual_dispatch_intent_output_and_is_json_safe():
    # End-to-end within this module: a real qa_engine.dispatch_intent()
    # grounded result (already JSON-safe) must flow straight into the
    # answer layer without any adaptation.
    import pandas as pd

    intent = {
        "intent": "period_value", "metric": None, "column_hint": "TotalPrice",
        "period_hint": "2024-03", "from_period_hint": None, "to_period_hint": None,
    }
    payload = {"numeric_summary": {"columns": [{"name": "TotalPrice", "count": 3, "missing_count": 0,
                                                 "sum": 450.0, "mean": 150.0, "median": 150.0,
                                                 "min": 100.0, "max": 200.0, "std": 50.0}]}}
    series = pd.Series([100.0, 200.0, 150.0], index=["2024-01", "2024-02", "2024-03"], name="TotalPrice")

    grounded_result = dispatch_intent(intent, payload, series)
    json.dumps(grounded_result)  # confirm JSON-safe before it reaches the answer layer

    answer = answer_grounded_result(grounded_result)

    assert answer == "TotalPrice in 2024-03 was 150.0."
