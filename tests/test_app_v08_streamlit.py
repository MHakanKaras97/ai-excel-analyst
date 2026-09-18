"""Streamlit AppTest coverage for the V0.8 UI additions: detected semantic
roles, baseline forecast, anomaly investigation, and evidence ("Why this
answer?") panels for both single-file Q&A and multi-file comparison.
"""
import json
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.gemini_provider import GeminiProvider

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def _monthly_excel_bytes(values):
    periods = pd.period_range(start="2023-01", periods=len(values), freq="M")
    df = pd.DataFrame({"period": [p.to_timestamp() for p in periods], "revenue": values})
    buffer = BytesIO()
    df.to_excel(buffer, index=False, engine="openpyxl")
    return buffer.getvalue()


def _upload_and_run(at, values):
    at.run()
    at.file_uploader[0].set_value(
        ("data.xlsx", _monthly_excel_bytes(values),
         "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    )
    at.run()
    return at


@pytest.fixture(autouse=True)
def _gemini_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")


def test_detected_semantic_roles_section_renders():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at = _upload_and_run(at, [100.0, 110.0, 120.0])

    assert not at.exception
    subheaders = [el.value for el in at.subheader]
    assert "Detected Semantic Roles" in subheaders
    file_id = next(iter(at.session_state["data_files"]))
    assert at.session_state["schema_results"][file_id]["columns"][0]["name"] == "period"


def test_forecast_caption_renders_for_sufficient_history():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at = _upload_and_run(at, [100.0, 110.0, 105.0, 120.0])

    assert not at.exception
    captions = [el.value for el in at.caption]
    assert any("Naive baseline forecast" in text for text in captions)


def test_forecast_caption_reports_unavailable_bounds_for_a_single_data_point():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at = _upload_and_run(at, [100.0])

    assert not at.exception
    captions = [el.value for el in at.caption]
    assert any("range unavailable" in text for text in captions)


def test_anomaly_investigation_expander_appears_when_anomalies_found():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    values = [100.0, 100.0, 100.0, 100.0, 500.0]
    at = _upload_and_run(at, values)

    assert not at.exception
    expander_labels = [el.label for el in at.expander]
    assert any("Investigate" in label for label in expander_labels)


def test_qa_evidence_panel_appears_after_valid_answer(monkeypatch):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at = _upload_and_run(at, [100.0, 110.0, 120.0])

    intent_response = json.dumps({
        "intent": "column_stat", "metric": "sum", "column_hint": "revenue",
        "period_hint": None, "from_period_hint": None, "to_period_hint": None,
    })
    monkeypatch.setattr(GeminiProvider, "generate", lambda self, prompt: intent_response)
    at.text_input(key="qa_question").set_value("What is total revenue?")
    at.run()
    at.button(key="qa_get_answer").click()
    at.run()

    assert not at.exception
    expander_labels = [el.label for el in at.expander]
    assert "Why this answer?" in expander_labels
    file_id = next(iter(at.session_state["data_files"]))
    assert st_evidence_has_no_file_id(at.session_state["qa_evidence"][file_id])


def st_evidence_has_no_file_id(evidence):
    return "file_id" not in json.dumps(evidence)


def test_explain_with_ai_button_produces_grounded_explanation(monkeypatch):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at = _upload_and_run(at, [100.0, 110.0, 120.0])

    intent_response = json.dumps({
        "intent": "column_stat", "metric": "sum", "column_hint": "revenue",
        "period_hint": None, "from_period_hint": None, "to_period_hint": None,
    })
    monkeypatch.setattr(GeminiProvider, "generate", lambda self, prompt: intent_response)
    at.text_input(key="qa_question").set_value("What is total revenue?")
    at.run()
    at.button(key="qa_get_answer").click()
    at.run()

    explanation_response = json.dumps({
        "summary": "The sum of revenue is 330.0.", "key_insights": [], "trend_interpretation": None,
        "warnings": [], "recommendations": [], "evidence_ids_used": ["ev_001"],
    })
    monkeypatch.setattr(GeminiProvider, "generate", lambda self, prompt: explanation_response)
    at.button(key="qa_explain_with_ai").click()
    at.run()

    assert not at.exception
    markdown_texts = [el.value for el in at.markdown]
    assert any("The sum of revenue is 330.0." in text for text in markdown_texts)


def test_qa_evidence_is_cleared_when_switching_active_file(monkeypatch):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at = _upload_and_run(at, [100.0, 110.0, 120.0])
    first_file_id = next(iter(at.session_state["data_files"]))

    intent_response = json.dumps({
        "intent": "column_stat", "metric": "sum", "column_hint": "revenue",
        "period_hint": None, "from_period_hint": None, "to_period_hint": None,
    })
    monkeypatch.setattr(GeminiProvider, "generate", lambda self, prompt: intent_response)
    at.text_input(key="qa_question").set_value("What is total revenue?")
    at.run()
    at.button(key="qa_get_answer").click()
    at.run()
    assert at.session_state["qa_evidence"][first_file_id] != []

    at.button(key="clear_all_files").click()
    at.run()

    assert at.session_state["qa_evidence"] == {}
    assert at.session_state["schema_results"] == {}


# ==================================================
# Multi-file comparison evidence panel
# ==================================================


def _revenue_excel_bytes(value):
    df = pd.DataFrame({
        "period": pd.to_datetime(["2024-01-01", "2024-02-01"]),
        "revenue": [value, value + 10.0],
    })
    buffer = BytesIO()
    df.to_excel(buffer, index=False, engine="openpyxl")
    return buffer.getvalue()


def test_comparison_evidence_panel_appears_after_valid_comparison_answer(monkeypatch):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    at.run()
    at.file_uploader[0].set_value([
        ("january.xlsx", _revenue_excel_bytes(100.0), mime),
        ("february.xlsx", _revenue_excel_bytes(200.0), mime),
    ])
    at.run()
    file_ids = list(at.session_state["data_files"].keys())
    at.selectbox(key=f"role_select_{file_ids[0]}").set_value("previous")
    at.selectbox(key=f"role_select_{file_ids[1]}").set_value("current")
    at.run()

    intent_response = json.dumps({
        "intent": "file_value_comparison", "column_hint": "revenue", "metric": "sum",
        "period_hint": None, "from_file_hint": "previous", "to_file_hint": "current",
    })
    monkeypatch.setattr(GeminiProvider, "generate", lambda self, prompt: intent_response)
    at.text_input(key="comparison_question").set_value("Compare revenue")
    at.run()
    at.button(key="comparison_get_answer").click()
    at.run()

    assert not at.exception
    expander_labels = [el.label for el in at.expander]
    assert "Why this answer?" in expander_labels
