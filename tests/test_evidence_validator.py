from src.evidence.validator import (
    contains_unsupported_causal_claim,
    validate_evidence_list,
    validate_referenced_evidence_ids,
)


def _evidence(evidence_id="ev_001", evidence_type="period_value"):
    return {"evidence_id": evidence_id, "type": evidence_type, "source": {"column": "Revenue"}, "value": 1}


def test_valid_evidence_list_passes():
    result = validate_evidence_list([_evidence()])

    assert result["valid"] is True


def test_duplicate_evidence_id_is_rejected():
    result = validate_evidence_list([_evidence(), _evidence()])

    assert result["valid"] is False
    assert result["reason"] == "duplicate_evidence_id"


def test_unknown_evidence_type_is_rejected():
    result = validate_evidence_list([_evidence(evidence_type="not_a_real_type")])

    assert result["valid"] is False
    assert result["reason"] == "unknown_evidence_type"


def test_malformed_evidence_entry_is_rejected():
    result = validate_evidence_list([{"not_evidence_id": "x"}])

    assert result["valid"] is False
    assert result["reason"] == "malformed_evidence"


def test_leaked_file_id_in_source_is_rejected():
    evidence = _evidence()
    evidence["source"] = {"file_id": "abc123", "column": "Revenue"}

    result = validate_evidence_list([evidence])

    assert result["valid"] is False
    assert result["reason"] == "file_id_leaked"


def test_leaked_file_id_nested_in_extra_field_is_rejected():
    evidence = _evidence()
    evidence["nested"] = {"file_a": {"file_id": "abc123"}}

    result = validate_evidence_list([evidence])

    assert result["valid"] is False
    assert result["reason"] == "file_id_leaked"


def test_empty_evidence_list_is_valid():
    assert validate_evidence_list([])["valid"] is True


# ==================================================
# validate_referenced_evidence_ids
# ==================================================


def test_known_referenced_ids_are_valid():
    result = validate_referenced_evidence_ids(["ev_001"], [_evidence()])

    assert result["valid"] is True
    assert result["unsupported_ids"] == []


def test_unknown_referenced_id_is_flagged():
    result = validate_referenced_evidence_ids(["ev_999"], [_evidence()])

    assert result["valid"] is False
    assert result["unsupported_ids"] == ["ev_999"]


def test_empty_referenced_ids_is_valid():
    result = validate_referenced_evidence_ids([], [_evidence()])

    assert result["valid"] is True


# ==================================================
# contains_unsupported_causal_claim
# ==================================================


def test_detects_caused_language():
    assert contains_unsupported_causal_claim("The decrease in units caused revenue to fall.") is True


def test_detects_led_to_language():
    assert contains_unsupported_causal_claim("The price change led to lower volume.") is True


def test_detects_due_to_the_decrease_language():
    assert contains_unsupported_causal_claim("Revenue fell due to the decrease in units sold.") is True


def test_neutral_coincidence_language_is_not_flagged():
    text = "Revenue decreased during the period, and units also decreased."
    assert contains_unsupported_causal_claim(text) is False


def test_empty_text_is_not_flagged():
    assert contains_unsupported_causal_claim("") is False


def test_case_insensitive_detection():
    assert contains_unsupported_causal_claim("This RESULTED IN a decline.") is True
