import pytest

from src.evidence.models import make_evidence, make_evidence_id


def test_make_evidence_returns_expected_shape():
    evidence = make_evidence("ev_001", "period_value", {"column": "Revenue", "period": "2025-03"}, value=125000)

    assert evidence == {
        "evidence_id": "ev_001", "type": "period_value",
        "source": {"column": "Revenue", "period": "2025-03"}, "value": 125000,
    }


def test_make_evidence_rejects_unknown_type():
    with pytest.raises(ValueError):
        make_evidence("ev_001", "not_a_real_type", {})


def test_make_evidence_rejects_empty_evidence_id():
    with pytest.raises(ValueError):
        make_evidence("", "period_value", {})


def test_make_evidence_rejects_file_id_in_source():
    with pytest.raises(ValueError):
        make_evidence("ev_001", "period_value", {"file_id": "abc123"})


def test_make_evidence_defaults_none_source_to_empty_dict():
    evidence = make_evidence("ev_001", "column_stat", None, value=1)

    assert evidence["source"] == {}


def test_make_evidence_id_zero_pads():
    assert make_evidence_id(1) == "ev_001"
    assert make_evidence_id(42) == "ev_042"
