from src.evidence.builder import build_evidence_from_comparison_result, build_evidence_from_grounded_result
from src.evidence.formatter import format_evidence, format_evidence_list


def test_format_period_value():
    result = {"found": True, "reason": None, "intent": "period_value", "column": "Revenue",
              "period": "2025-03", "value": 125000, "extra": {}}
    evidence = build_evidence_from_grounded_result(result)[0]

    assert format_evidence(evidence) == "Revenue in 2025-03 was 125000."


def test_format_period_change_includes_percentage():
    result = {
        "found": True, "reason": None, "intent": "period_change", "column": "Revenue", "period": None,
        "value": -25000,
        "extra": {"from_period": "2025-02", "to_period": "2025-03", "previous": 150000,
                  "current": 125000, "absolute_change": -25000, "percentage_change": -16.67, "is_valid": True},
    }
    evidence = build_evidence_from_grounded_result(result)[0]

    text = format_evidence(evidence)

    assert "150000" in text and "125000" in text and "-25000" in text and "-16.67" in text


def test_format_column_stat():
    result = {"found": True, "reason": None, "intent": "column_stat", "column": "Revenue",
              "period": None, "value": 460000.0, "extra": {"metric": "sum"}}
    evidence = build_evidence_from_grounded_result(result)[0]

    assert format_evidence(evidence) == "The sum of Revenue is 460000.0."


def test_format_anomaly_summary_zero_anomalies():
    result = {
        "found": True, "reason": None, "intent": "anomaly_check", "column": "Revenue", "period": None, "value": 0,
        "extra": {"insufficient_data": False, "lower_bound": 1.0, "upper_bound": 2.0, "anomalies": []},
    }
    evidence = build_evidence_from_grounded_result(result)[0]

    assert format_evidence(evidence) == "No anomalies were detected for Revenue."


def test_format_file_comparison():
    result = {
        "found": True, "reason": None, "comparison_type": "file_value_comparison", "metric": "sum",
        "column": "Revenue", "period": None,
        "file_a": {"file_id": "abc", "display_name": "january.xlsx", "role": "previous"},
        "file_b": {"file_id": "def", "display_name": "february.xlsx", "role": "current"},
        "previous": 210.0, "current": 410.0, "absolute_change": 200.0, "percentage_change": 95.24,
        "is_valid": True, "extra": {},
    }
    evidence = build_evidence_from_comparison_result(result)[0]

    text = format_evidence(evidence)

    assert "january.xlsx" in text and "february.xlsx" in text and "210.0" in text and "410.0" in text


def test_unknown_evidence_type_falls_back_to_generic_message():
    evidence = {"evidence_id": "ev_001", "type": "trend", "source": {"column": "Revenue"}, "trend": "increasing"}

    assert format_evidence(evidence) == "Revenue's trend is increasing."


def test_format_evidence_list_formats_each_item():
    result = {"found": True, "reason": None, "intent": "column_stat", "column": "Revenue",
              "period": None, "value": 1.0, "extra": {"metric": "sum"}}
    evidence_list = build_evidence_from_grounded_result(result)

    formatted = format_evidence_list(evidence_list)

    assert formatted == ["The sum of Revenue is 1.0."]


def test_format_evidence_list_empty_input():
    assert format_evidence_list([]) == []
