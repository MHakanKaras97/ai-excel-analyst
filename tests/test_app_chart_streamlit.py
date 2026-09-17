"""Streamlit AppTest coverage for the V0.6.6 chart-question section.

Mirrors tests/test_app_qa_streamlit.py's conventions exactly: drives the
real app.py through streamlit.testing.v1.AppTest, simulating a file upload
and widget interactions, and mocks the Gemini SDK boundary by patching
GeminiProvider.generate directly (the class object is shared across every
`from src.gemini_provider import GeminiProvider` import, including the one
AppTest re-execs inside app.py) — no live Gemini call is made.
"""
import json
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.gemini_provider import GeminiProvider

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def _synthetic_excel_bytes():
    df = pd.DataFrame({
        "period": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01"]),
        "revenue": ["$100,000", "$110,000", "$120,000", "$130,000"],
    })
    buffer = BytesIO()
    df.to_excel(buffer, index=False, engine="openpyxl")
    return buffer.getvalue()


def _upload_and_run(at):
    at.run()
    at.file_uploader[0].set_value(
        ("quarterly_revenue.xlsx", _synthetic_excel_bytes(),
         "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    )
    at.run()
    return at


def _ask_chart(at, question: str, provider_response: str, monkeypatch):
    monkeypatch.setattr(GeminiProvider, "generate", lambda self, prompt: provider_response)
    at.text_input(key="chart_question").set_value(question)
    at.run()
    at.button(key="chart_show_chart").click()
    at.run()
    return at


def _ask_qa(at, question: str, provider_response: str, monkeypatch):
    monkeypatch.setattr(GeminiProvider, "generate", lambda self, prompt: provider_response)
    at.text_input(key="qa_question").set_value(question)
    at.run()
    at.button(key="qa_get_answer").click()
    at.run()
    return at


@pytest.fixture(autouse=True)
def _gemini_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")


def _chart_intent_response(**overrides):
    payload = {
        "intent": "trend_chart", "column_hint": None, "metric": None,
        "period_hint": None, "from_period_hint": None, "to_period_hint": None,
    }
    payload.update(overrides)
    return json.dumps(payload)


# ==================================================
# SUCCESSFUL CHART REQUESTS
# ==================================================


def test_successful_trend_chart_request_renders_plotly_chart(monkeypatch):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at = _upload_and_run(at)

    at = _ask_chart(at, "Show me the monthly revenue trend.",
                     _chart_intent_response(intent="trend_chart", column_hint="revenue"), monkeypatch)

    assert not at.exception
    assert len(at.get("plotly_chart")) >= 1


def test_successful_numeric_summary_chart_request_renders_plotly_chart(monkeypatch):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at = _upload_and_run(at)

    before = len(at.get("plotly_chart"))
    at = _ask_chart(at, "Show the average revenue.",
                     _chart_intent_response(intent="numeric_summary_chart", column_hint="revenue", metric="mean"),
                     monkeypatch)

    assert not at.exception
    assert len(at.get("plotly_chart")) > before


def test_successful_period_change_chart_request_renders_plotly_chart(monkeypatch):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at = _upload_and_run(at)

    before = len(at.get("plotly_chart"))
    at = _ask_chart(at, "Show the change from January to February.",
                     _chart_intent_response(intent="period_change_chart",
                                             from_period_hint="January", to_period_hint="February"),
                     monkeypatch)

    assert not at.exception
    assert len(at.get("plotly_chart")) > before


def test_successful_missing_values_chart_request_renders_plotly_chart(monkeypatch):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at = _upload_and_run(at)

    before = len(at.get("plotly_chart"))
    at = _ask_chart(at, "Show missing values.",
                     _chart_intent_response(intent="missing_values_chart", column_hint=None), monkeypatch)

    assert not at.exception
    assert len(at.get("plotly_chart")) > before


# ==================================================
# FAILURE PATHS
# ==================================================


def test_interpreter_failure_renders_no_chart_and_shows_error(monkeypatch):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at = _upload_and_run(at)

    before = len(at.get("plotly_chart"))
    at = _ask_chart(at, "Show the trend.", "not valid json", monkeypatch)

    assert not at.exception
    assert len(at.get("plotly_chart")) == before
    assert any("Could not understand the chart request." in el.value for el in at.error)


def test_engine_failure_renders_no_chart_and_shows_error(monkeypatch):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at = _upload_and_run(at)

    before = len(at.get("plotly_chart"))
    at = _ask_chart(at, "Show the Nonexistent trend.",
                     _chart_intent_response(intent="trend_chart", column_hint="Nonexistent"), monkeypatch)

    assert not at.exception
    assert len(at.get("plotly_chart")) == before
    assert any("column was not found" in el.value for el in at.error)


def test_unsupported_chart_request_renders_no_chart_with_clear_feedback(monkeypatch):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at = _upload_and_run(at)

    before = len(at.get("plotly_chart"))
    at = _ask_chart(at, "Why did revenue decrease?",
                     _chart_intent_response(intent="unsupported", column_hint=None), monkeypatch)

    assert not at.exception
    assert len(at.get("plotly_chart")) == before
    assert any("isn't supported yet" in el.value for el in at.error)


def test_blank_chart_question_does_not_error():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at = _upload_and_run(at)

    at.button(key="chart_show_chart").click()
    at.run()

    assert not at.exception


# ==================================================
# EXISTING BEHAVIOR PRESERVED
# ==================================================


def test_existing_dashboard_behavior_remains_intact():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at = _upload_and_run(at)

    assert not at.exception
    subheaders = [el.value for el in at.subheader]
    assert "Dataset Information" in subheaders
    assert "Deterministic Analytics" in subheaders
    assert "Trend Analysis" in subheaders
    assert "AI-Generated Insight" in subheaders
    assert "Ask a Question" in subheaders
    assert "Ask for a Chart" in subheaders


def test_existing_qa_behavior_remains_intact(monkeypatch):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at = _upload_and_run(at)

    intent_response = json.dumps({
        "intent": "column_stat", "metric": "sum", "column_hint": "revenue",
        "period_hint": None, "from_period_hint": None, "to_period_hint": None,
    })
    at = _ask_qa(at, "What is the total revenue?", intent_response, monkeypatch)

    assert not at.exception
    answer_texts = [el.value for el in at.markdown]
    assert any("460000.0" in text and "sum" in text and "revenue" in text for text in answer_texts)


# ==================================================
# PROVIDER CALL COUNT
# ==================================================


def test_provider_generate_called_exactly_once_per_chart_question(monkeypatch):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at = _upload_and_run(at)

    call_count = {"n": 0}

    def fake_generate(self, prompt):
        call_count["n"] += 1
        return _chart_intent_response(intent="trend_chart", column_hint="revenue")

    monkeypatch.setattr(GeminiProvider, "generate", fake_generate)

    at.text_input(key="chart_question").set_value("Show me the monthly revenue trend.")
    at.run()
    at.button(key="chart_show_chart").click()
    at.run()

    assert not at.exception
    assert call_count["n"] == 1
