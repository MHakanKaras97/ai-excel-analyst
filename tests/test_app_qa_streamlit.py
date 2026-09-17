"""Streamlit AppTest coverage for the V0.5.6 Q&A section.

Drives the real app.py through streamlit.testing.v1.AppTest, simulating a
file upload and widget interactions. The Gemini SDK boundary is mocked by
patching GeminiProvider.generate directly (the class object is shared across
every `from src.gemini_provider import GeminiProvider` import, including the
one AppTest re-execs inside app.py) — no live Gemini call is made.
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


def _ask(at, question: str, provider_response: str, monkeypatch):
    monkeypatch.setattr(GeminiProvider, "generate", lambda self, prompt: provider_response)
    at.text_input(key="qa_question").set_value(question)
    at.run()
    at.button(key="qa_get_answer").click()
    at.run()
    return at


@pytest.fixture(autouse=True)
def _gemini_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")


def test_existing_dashboard_smoke_test_still_passes():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception

    at = _upload_and_run(at)
    assert not at.exception

    subheaders = [el.value for el in at.subheader]
    assert "Dataset Information" in subheaders
    assert "Deterministic Analytics" in subheaders
    assert "Trend Analysis" in subheaders
    assert "AI-Generated Insight" in subheaders


def test_qa_section_renders():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at = _upload_and_run(at)

    assert not at.exception
    subheaders = [el.value for el in at.subheader]
    assert "Ask a Question" in subheaders
    assert any(w.key == "qa_question" for w in at.text_input)
    assert any(w.key == "qa_get_answer" for w in at.button)


def test_valid_question_produces_deterministic_answer(monkeypatch):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at = _upload_and_run(at)

    intent_response = json.dumps({
        "intent": "column_stat", "metric": "sum", "column_hint": "revenue",
        "period_hint": None, "from_period_hint": None, "to_period_hint": None,
    })
    at = _ask(at, "What is the total revenue?", intent_response, monkeypatch)

    assert not at.exception
    answer_texts = [el.value for el in at.markdown]
    assert any("460000.0" in text and "sum" in text and "revenue" in text for text in answer_texts)


def test_unsupported_question_produces_deterministic_response(monkeypatch):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at = _upload_and_run(at)

    intent_response = json.dumps({
        "intent": "unsupported", "metric": None, "column_hint": None,
        "period_hint": None, "from_period_hint": None, "to_period_hint": None,
    })
    at = _ask(at, "What's the weather like today?", intent_response, monkeypatch)

    assert not at.exception
    answer_texts = [el.value for el in at.markdown]
    assert "That question isn't supported yet." in answer_texts


def test_malformed_llm_response_produces_deterministic_response_without_crashing(monkeypatch):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at = _upload_and_run(at)

    at = _ask(at, "What is the total revenue?", "not valid json", monkeypatch)

    assert not at.exception
    answer_texts = [el.value for el in at.markdown]
    assert "The question couldn't be interpreted (the AI response wasn't valid JSON)." in answer_texts


def test_blank_question_does_not_error():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at = _upload_and_run(at)

    at.button(key="qa_get_answer").click()
    at.run()

    assert not at.exception
