"""End-to-end integration tests spanning multiple V0.8 modules together,
as distinct from each module's own focused unit tests:

- semantic proposal -> validation
- analysis -> evidence

("Q&A -> evidence -> AI explanation -> validation" is covered by
tests/test_evidence_qa.py, and "forecast -> evaluation" by
tests/test_forecast_evaluator.py — both already exercise their own
full pipelines end to end.)
"""
import json

import pandas as pd

from src.evidence.builder import build_evidence_from_grounded_result
from src.evidence.validator import validate_evidence_list
from src.file_analysis import analyze_file
from src.qa_engine import dispatch_intent
from src.schema.schema_analyzer import analyze_schema
from src.schema.schema_resolver import apply_semantic_proposals, build_candidate_column_metadata
from src.schema.semantic_interpreter import interpret_semantic_schema


class FakeProvider:
    def __init__(self, response):
        self._response = response
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        self.call_count += 1
        return self._response


# ==================================================
# semantic proposal -> validation
# ==================================================


def test_semantic_proposal_flows_through_to_a_validated_dataset_schema():
    df = pd.DataFrame({"Discount": ["10%", "12%", "15%"], "Region": ["North", "South", "East"]})
    candidate_schema = analyze_schema(df)
    metadata = build_candidate_column_metadata(candidate_schema)

    llm_response = json.dumps({"columns": [
        {"name": "Discount", "role": "measure", "semantic_type": "percentage", "unit": "%", "time_role": "none"},
        {"name": "Region", "role": "dimension", "semantic_type": "text", "unit": None, "time_role": "none"},
    ]})
    provider = FakeProvider(llm_response)

    interpretation = interpret_semantic_schema(metadata, provider)
    assert interpretation["is_valid"] is True

    merged = apply_semantic_proposals(candidate_schema, interpretation["proposals"])

    discount = next(c for c in merged["schema"]["columns"] if c["name"] == "Discount")
    assert discount["semantic_type"] == "percentage"
    assert discount["unit"] == "%"


def test_semantic_proposal_contradicting_strong_evidence_is_overridden_end_to_end():
    df = pd.DataFrame({"Revenue": [100.0, 200.0, 300.0]})
    candidate_schema = analyze_schema(df)
    metadata = build_candidate_column_metadata(candidate_schema)

    # The LLM incorrectly proposes "dimension" for a dtype-certain numeric
    # measure column — Python must reject this, not accept it verbatim.
    llm_response = json.dumps({"columns": [
        {"name": "Revenue", "role": "dimension", "semantic_type": "text", "unit": None, "time_role": "none"},
    ]})
    provider = FakeProvider(llm_response)

    interpretation = interpret_semantic_schema(metadata, provider)
    merged = apply_semantic_proposals(candidate_schema, interpretation["proposals"])

    revenue = next(c for c in merged["schema"]["columns"] if c["name"] == "Revenue")
    assert revenue["role"] == "measure"
    assert merged["decisions"]["Revenue"][0]["accepted"] is False


# ==================================================
# analysis -> evidence
# ==================================================


def test_file_analysis_flows_through_qa_engine_into_valid_evidence():
    df = pd.DataFrame({
        "period": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
        "Revenue": [100.0, 110.0, 121.0],
    })
    analysis = analyze_file(df, value_position=1, period_position=0)

    intent = {"intent": "column_stat", "metric": "sum", "column_hint": "Revenue",
              "period_hint": None, "from_period_hint": None, "to_period_hint": None}
    grounded_result = dispatch_intent(intent, analysis, analysis["monthly_series"])
    assert grounded_result["found"] is True

    evidence = build_evidence_from_grounded_result(grounded_result, file_label="report.xlsx")

    assert validate_evidence_list(evidence)["valid"] is True
    assert evidence[0]["value"] == 331.0
    assert evidence[0]["source"]["file"] == "report.xlsx"
    assert "file_id" not in json.dumps(evidence)


def test_file_analysis_period_change_flows_into_valid_evidence():
    df = pd.DataFrame({
        "period": pd.to_datetime(["2024-01-01", "2024-02-01"]),
        "Revenue": [150000.0, 125000.0],
    })
    analysis = analyze_file(df, value_position=1, period_position=0)

    intent = {"intent": "period_change", "metric": None, "column_hint": "Revenue",
              "period_hint": None, "from_period_hint": "2024-01", "to_period_hint": "2024-02"}
    grounded_result = dispatch_intent(intent, analysis, analysis["monthly_series"])
    assert grounded_result["found"] is True

    evidence = build_evidence_from_grounded_result(grounded_result)

    assert validate_evidence_list(evidence)["valid"] is True
    assert evidence[0]["previous"] == 150000.0
    assert evidence[0]["current"] == 125000.0
