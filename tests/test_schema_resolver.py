import pytest

from src.schema.schema_analyzer import analyze_schema
from src.schema.schema_resolver import (
    apply_semantic_proposals,
    build_candidate_column_metadata,
    merge_llm_schema_proposal,
    resolve_semantic_column,
)


def _columns(*names):
    return [{"name": name} for name in names]


# ==================================================
# resolve_semantic_column
# ==================================================


def test_exact_match_is_used_before_semantic_scoring():
    result = resolve_semantic_column("Revenue", _columns("Revenue", "Net Revenue"))

    assert result["found"] is True
    assert result["column"] == "Revenue"
    assert result["confidence"] == 1.0


def test_exact_case_insensitive_ambiguity_is_reported_as_is():
    result = resolve_semantic_column("revenue", _columns("Revenue", "REVENUE"))

    assert result["found"] is False
    assert result["reason"] == "ambiguous_column"
    assert set(result["candidates"]) == {"Revenue", "REVENUE"}


def test_semantic_term_resolves_uniquely_to_high_confidence_candidate():
    result = resolve_semantic_column("total sales", _columns("Sales Amount", "Net Revenue", "Units Sold"))

    assert result["found"] is True
    assert result["column"] == "Sales Amount"
    assert result["confidence"] >= 0.6


def test_semantic_term_with_no_plausible_candidate_is_not_found():
    result = resolve_semantic_column("shipping weight", _columns("Sales Amount", "Region"))

    assert result["found"] is False
    assert result["reason"] == "column_not_found"


def test_blank_hint_is_not_found():
    result = resolve_semantic_column("   ", _columns("Sales Amount"))

    assert result["found"] is False
    assert result["reason"] == "column_not_found"


def test_semantic_term_matching_two_similarly_scored_candidates_is_ambiguous():
    result = resolve_semantic_column("net", _columns("Net Revenue", "Net Profit"))

    assert result["found"] is False
    assert result["reason"] == "ambiguous_column"
    returned_columns = {c["column"] for c in result["candidates"]}
    assert returned_columns == {"Net Revenue", "Net Profit"}


def test_high_margin_between_two_high_confidence_candidates_is_still_ambiguous_by_design():
    # Documents the intentional "resolve don't guess" trade-off: ambiguity
    # is a count check (how many candidates clear the high-confidence
    # band), not a margin check. "Gross Profit Margin" (1.0) clearly
    # outscores "Profit" (0.667), yet both still clear HIGH_CONFIDENCE with
    # real literal overlap, so neither is auto-picked.
    result = resolve_semantic_column("gross profit", _columns("Gross Profit Margin", "Profit"))

    assert result["found"] is False
    assert result["reason"] == "ambiguous_column"
    scores = {c["column"]: c["confidence"] for c in result["candidates"]}
    assert scores["Gross Profit Margin"] == 1.0
    assert scores["Profit"] == pytest.approx(0.6667, abs=1e-3)
    assert scores["Gross Profit Margin"] - scores["Profit"] > 0.3  # a large margin, still ambiguous


def test_single_medium_confidence_candidate_is_ambiguous_not_resolved():
    # "amount" alone only weakly overlaps "Total Price" (no shared synonym
    # group covering "price"/"amount"), so nothing should auto-resolve.
    result = resolve_semantic_column("amount", _columns("Total Price"))

    assert result["found"] is False


def test_confidence_score_never_exceeds_one():
    # "margin" expands (via the profit/margin synonym group) to overlap
    # BOTH tokens of "Profit Margin" despite being a single-word hint —
    # a naive Dice calculation using the unexpanded hint length as the
    # denominator would push the score above 1.0.
    result = resolve_semantic_column("margin", _columns("Profit Margin", "Region"))

    assert result["found"] is True
    assert result["confidence"] <= 1.0


def test_does_not_mutate_schema_columns_input():
    columns = _columns("Sales Amount", "Net Revenue")
    before = [dict(c) for c in columns]

    resolve_semantic_column("total sales", columns)

    assert columns == before


# ==================================================
# merge_llm_schema_proposal
# ==================================================


def _candidate(**overrides):
    base = {
        "name": "Discount", "role": "unknown", "semantic_type": "unknown",
        "unit": None, "time_role": "none", "confidence": 0.3, "evidence": {},
    }
    base.update(overrides)
    return base


def test_valid_proposal_is_accepted_when_candidate_confidence_is_low():
    candidate = _candidate()
    proposal = {"role": "measure", "semantic_type": "percentage", "unit": "%"}

    result = merge_llm_schema_proposal(candidate, proposal)

    assert result["schema"]["role"] == "measure"
    assert result["schema"]["semantic_type"] == "percentage"
    assert result["schema"]["unit"] == "%"
    assert all(d["accepted"] for d in result["decisions"])


def test_invalid_role_is_rejected():
    candidate = _candidate()
    proposal = {"role": "not_a_real_role"}

    result = merge_llm_schema_proposal(candidate, proposal)

    assert result["schema"]["role"] == "unknown"
    assert result["decisions"] == [{"accepted": False, "field": "role", "reason": "invalid_value"}]


def test_invalid_semantic_type_is_rejected():
    candidate = _candidate()
    proposal = {"semantic_type": "not_a_real_type"}

    result = merge_llm_schema_proposal(candidate, proposal)

    assert result["schema"]["semantic_type"] == "unknown"
    assert result["decisions"][0]["accepted"] is False


def test_proposal_contradicting_strong_deterministic_evidence_is_rejected():
    candidate = _candidate(role="measure", semantic_type="numeric", confidence=0.9)
    proposal = {"role": "dimension"}

    result = merge_llm_schema_proposal(candidate, proposal)

    assert result["schema"]["role"] == "measure"
    assert result["decisions"] == [{"accepted": False, "field": "role", "reason": "contradicts_strong_evidence"}]


def test_proposal_agreeing_with_strong_evidence_is_accepted():
    candidate = _candidate(role="measure", semantic_type="numeric", confidence=0.9)
    proposal = {"role": "measure"}

    result = merge_llm_schema_proposal(candidate, proposal)

    assert result["schema"]["role"] == "measure"
    assert result["decisions"] == [{"accepted": True, "field": "role", "value": "measure"}]


def test_unit_proposal_contradicting_strong_evidence_is_rejected():
    # Regression test: unit must be gated by the same strong-evidence rule
    # as role/semantic_type/time_role — a dtype-certain column's unit is
    # never silently overridden just because unit has no fixed enum.
    candidate = _candidate(role="measure", semantic_type="numeric", confidence=0.9, unit=None)
    proposal = {"unit": "USD"}

    result = merge_llm_schema_proposal(candidate, proposal)

    assert result["schema"]["unit"] is None
    assert result["decisions"] == [{"accepted": False, "field": "unit", "reason": "contradicts_strong_evidence"}]


def test_unit_proposal_agreeing_with_strong_evidence_is_accepted():
    candidate = _candidate(role="measure", semantic_type="currency", confidence=0.9, unit="USD")
    proposal = {"unit": "USD"}

    result = merge_llm_schema_proposal(candidate, proposal)

    assert result["schema"]["unit"] == "USD"
    assert result["decisions"] == [{"accepted": True, "field": "unit", "value": "USD"}]


def test_unit_proposal_is_freely_settable_on_low_confidence_candidate():
    candidate = _candidate(unit=None, confidence=0.3)
    proposal = {"unit": "EUR"}

    result = merge_llm_schema_proposal(candidate, proposal)

    assert result["schema"]["unit"] == "EUR"
    assert result["decisions"] == [{"accepted": True, "field": "unit", "value": "EUR"}]


def test_missing_proposal_fields_are_left_untouched():
    candidate = _candidate(unit="USD")

    result = merge_llm_schema_proposal(candidate, {})

    assert result["schema"] == candidate
    assert result["decisions"] == []


def test_invalid_unit_type_is_rejected():
    candidate = _candidate()
    proposal = {"unit": 123}

    result = merge_llm_schema_proposal(candidate, proposal)

    assert result["schema"]["unit"] is None
    assert result["decisions"][0] == {"accepted": False, "field": "unit", "reason": "invalid_value"}


def test_does_not_mutate_candidate_column():
    candidate = _candidate()
    before = dict(candidate)

    merge_llm_schema_proposal(candidate, {"role": "measure"})

    assert candidate == before


def test_invalid_time_role_is_rejected():
    candidate = _candidate()
    proposal = {"time_role": "not_a_real_time_role"}

    result = merge_llm_schema_proposal(candidate, proposal)

    assert result["schema"]["time_role"] == "none"
    assert result["decisions"][0]["accepted"] is False


# ==================================================
# build_candidate_column_metadata / apply_semantic_proposals
# ==================================================


def test_build_candidate_column_metadata_excludes_internal_fields():
    import pandas as pd

    df = pd.DataFrame({"Discount": ["10%", "12%"]})
    schema = analyze_schema(df)

    metadata = build_candidate_column_metadata(schema)

    assert metadata[0]["name"] == "Discount"
    assert "confidence" not in metadata[0]
    assert metadata[0]["candidate_semantic_type"] == "percentage"
    assert "file_id" not in metadata[0]


def test_apply_semantic_proposals_updates_matching_columns_only():
    dataset_schema = {"columns": [
        _candidate(name="Discount"),
        _candidate(name="Region", role="dimension", semantic_type="text", confidence=0.65),
    ]}
    proposals = {"Discount": {"role": "measure", "semantic_type": "percentage", "unit": "%", "time_role": "none"}}

    result = apply_semantic_proposals(dataset_schema, proposals)

    discount = next(c for c in result["schema"]["columns"] if c["name"] == "Discount")
    region = next(c for c in result["schema"]["columns"] if c["name"] == "Region")
    assert discount["semantic_type"] == "percentage"
    assert region == _candidate(name="Region", role="dimension", semantic_type="text", confidence=0.65)
    assert "Discount" in result["decisions"]
    assert "Region" not in result["decisions"]


def test_apply_semantic_proposals_does_not_mutate_input_schema():
    dataset_schema = {"columns": [_candidate(name="Discount")]}
    before = {"columns": [dict(c) for c in dataset_schema["columns"]]}

    apply_semantic_proposals(dataset_schema, {"Discount": {"role": "measure"}})

    assert dataset_schema == before
