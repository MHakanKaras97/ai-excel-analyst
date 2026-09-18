from src.evidence.builder import build_evidence_from_comparison_result, build_evidence_from_grounded_result


def test_builds_period_value_evidence():
    result = {"found": True, "reason": None, "intent": "period_value", "column": "Revenue",
              "period": "2025-03", "value": 125000, "extra": {}}

    evidence = build_evidence_from_grounded_result(result, file_label="sales.xlsx")

    assert evidence == [{
        "evidence_id": "ev_001", "type": "period_value",
        "source": {"file": "sales.xlsx", "column": "Revenue", "period": "2025-03"},
        "value": 125000,
    }]


def test_builds_period_change_evidence():
    result = {
        "found": True, "reason": None, "intent": "period_change", "column": "Revenue", "period": None,
        "value": -25000,
        "extra": {"from_period": "2025-02", "to_period": "2025-03", "previous": 150000,
                  "current": 125000, "absolute_change": -25000, "percentage_change": -16.67, "is_valid": True},
    }

    evidence = build_evidence_from_grounded_result(result)

    assert evidence[0]["type"] == "period_change"
    assert evidence[0]["previous"] == 150000
    assert evidence[0]["current"] == 125000
    assert evidence[0]["absolute_change"] == -25000
    assert evidence[0]["percentage_change"] == -16.67
    assert evidence[0]["source"] == {"column": "Revenue", "from_period": "2025-02", "to_period": "2025-03"}


def test_period_change_evidence_propagates_is_valid():
    result = {
        "found": True, "reason": None, "intent": "period_change", "column": "Revenue", "period": None,
        "value": None,
        "extra": {"from_period": "2025-02", "to_period": "2025-03", "previous": 0,
                  "current": 50.0, "absolute_change": 50.0, "percentage_change": None, "is_valid": False},
    }

    evidence = build_evidence_from_grounded_result(result)

    assert evidence[0]["is_valid"] is False


def test_builds_column_stat_evidence():
    result = {"found": True, "reason": None, "intent": "column_stat", "column": "Revenue",
              "period": None, "value": 460000.0, "extra": {"metric": "sum"}}

    evidence = build_evidence_from_grounded_result(result)

    assert evidence[0]["type"] == "column_stat"
    assert evidence[0]["value"] == 460000.0
    assert evidence[0]["metric"] == "sum"


def test_builds_anomaly_summary_evidence():
    result = {
        "found": True, "reason": None, "intent": "anomaly_check", "column": "Revenue", "period": None, "value": 1,
        "extra": {"insufficient_data": False, "lower_bound": 10.0, "upper_bound": 190.0,
                  "anomalies": [{"position": 1, "period": "2024-02", "value": 200.0, "direction": "high"}]},
    }

    evidence = build_evidence_from_grounded_result(result)

    assert evidence[0]["type"] == "anomaly_summary"
    assert evidence[0]["anomaly_count"] == 1
    assert evidence[0]["anomalies"][0]["direction"] == "high"


def test_builds_missing_values_evidence():
    result = {"found": True, "reason": None, "intent": "missing_values", "column": None, "period": None,
              "value": 1, "extra": {"columns": [{"name": "Region", "missing_count": 2, "missing_percentage": 20.0}]}}

    evidence = build_evidence_from_grounded_result(result)

    assert evidence[0]["type"] == "missing_values"
    assert evidence[0]["column_count_with_missing"] == 1


def test_failed_grounded_result_produces_no_evidence():
    result = {"found": False, "reason": "column_not_found", "intent": "column_stat",
              "column": None, "period": None, "value": None, "extra": {}}

    assert build_evidence_from_grounded_result(result) == []


def test_unsupported_intent_produces_no_evidence():
    result = {"found": False, "reason": "unsupported", "intent": "unsupported",
              "column": None, "period": None, "value": None, "extra": {}}

    assert build_evidence_from_grounded_result(result) == []


def test_evidence_never_contains_file_id():
    result = {"found": True, "reason": None, "intent": "column_stat", "column": "Revenue",
              "period": None, "value": 1.0, "extra": {"metric": "sum"}}

    evidence = build_evidence_from_grounded_result(result, file_label="Q1.xlsx")

    assert "file_id" not in evidence[0]["source"]


def test_start_index_controls_generated_evidence_id():
    result = {"found": True, "reason": None, "intent": "column_stat", "column": "Revenue",
              "period": None, "value": 1.0, "extra": {"metric": "sum"}}

    evidence = build_evidence_from_grounded_result(result, start_index=5)

    assert evidence[0]["evidence_id"] == "ev_005"


def test_does_not_mutate_grounded_result():
    result = {"found": True, "reason": None, "intent": "column_stat", "column": "Revenue",
              "period": None, "value": 1.0, "extra": {"metric": "sum"}}
    before = {**result, "extra": dict(result["extra"])}

    build_evidence_from_grounded_result(result)

    assert result == before


# ==================================================
# build_evidence_from_comparison_result
# ==================================================


def test_builds_file_comparison_evidence():
    result = {
        "found": True, "reason": None, "comparison_type": "file_value_comparison", "metric": "sum",
        "column": "Revenue", "period": None,
        "file_a": {"file_id": "abc", "display_name": "january.xlsx", "role": "previous"},
        "file_b": {"file_id": "def", "display_name": "february.xlsx", "role": "current"},
        "previous": 210.0, "current": 410.0, "absolute_change": 200.0, "percentage_change": 95.24,
        "is_valid": True, "extra": {},
    }

    evidence = build_evidence_from_comparison_result(result)

    assert evidence[0]["type"] == "file_comparison"
    assert evidence[0]["previous"] == 210.0
    assert evidence[0]["current"] == 410.0
    assert evidence[0]["source"]["file_a"] == "january.xlsx"
    assert evidence[0]["source"]["file_b"] == "february.xlsx"
    assert evidence[0]["is_valid"] is True


def test_file_comparison_evidence_propagates_is_valid_and_compare_reason():
    result = {
        "found": True, "reason": None, "comparison_type": "file_value_comparison", "metric": "sum",
        "column": "Revenue", "period": None,
        "file_a": {"file_id": "abc", "display_name": "january.xlsx", "role": "previous"},
        "file_b": {"file_id": "def", "display_name": "february.xlsx", "role": "current"},
        "previous": 0.0, "current": 50.0, "absolute_change": 50.0, "percentage_change": None,
        "is_valid": False, "extra": {"compare_reason": "division_by_zero"},
    }

    evidence = build_evidence_from_comparison_result(result)

    assert evidence[0]["is_valid"] is False
    assert evidence[0]["compare_reason"] == "division_by_zero"


def test_comparison_evidence_never_leaks_file_id():
    result = {
        "found": True, "reason": None, "comparison_type": "file_value_comparison", "metric": "sum",
        "column": "Revenue", "period": None,
        "file_a": {"file_id": "abc", "display_name": "january.xlsx", "role": "previous"},
        "file_b": {"file_id": "def", "display_name": "february.xlsx", "role": "current"},
        "previous": 210.0, "current": 410.0, "absolute_change": 200.0, "percentage_change": 95.24,
        "is_valid": True, "extra": {},
    }

    evidence = build_evidence_from_comparison_result(result)

    assert "file_id" not in evidence[0]["source"]
    assert "abc" not in str(evidence[0])
    assert "def" not in str(evidence[0])


def test_failed_comparison_result_produces_no_evidence():
    result = {"found": False, "reason": "file_not_found", "comparison_type": "file_value_comparison",
              "metric": None, "column": None, "period": None, "file_a": None, "file_b": None,
              "previous": None, "current": None, "absolute_change": None, "percentage_change": None,
              "is_valid": None, "extra": {}}

    assert build_evidence_from_comparison_result(result) == []
