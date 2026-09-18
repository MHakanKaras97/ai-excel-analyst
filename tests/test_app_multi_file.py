"""Tests for V0.7.1: DataFile registration and multi-file session-state
foundation. No per-file analysis, comparison, or roles-based logic exists
yet — that is V0.7.2+.
"""
import json
import uuid
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app import determine_active_file_id, register_uploaded_file
from src.gemini_provider import GeminiProvider

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


class FakeUploadedFile:
    """Mirrors tests/test_app_helpers.py's double, plus an optional
    Streamlit-style `file_id` for testing the rerun-dedup path."""

    def __init__(self, name: str, content: bytes, file_id: str | None = None):
        self.name = name
        self._content = content
        if file_id is not None:
            self.file_id = file_id

    def getbuffer(self):
        return self._content


def _excel_bytes(value=100.0):
    df = pd.DataFrame({"period": pd.to_datetime(["2024-01-01"]), "revenue": [value]})
    buffer = BytesIO()
    df.to_excel(buffer, index=False, engine="openpyxl")
    return buffer.getvalue()


# ==================================================
# register_uploaded_file — unit tests
# ==================================================


def test_register_uploaded_file_returns_expected_shape():
    uploaded = FakeUploadedFile("report.xlsx", _excel_bytes())

    data_file = register_uploaded_file(uploaded)

    assert set(data_file.keys()) >= {
        "file_id", "filename", "display_name", "role",
        "raw_df", "value_column", "period_column", "analysis",
    }
    assert isinstance(data_file["raw_df"], pd.DataFrame)
    assert list(data_file["raw_df"].columns) == ["period", "revenue"]


def test_register_uploaded_file_generates_uuid_file_id():
    uploaded = FakeUploadedFile("report.xlsx", _excel_bytes())

    data_file = register_uploaded_file(uploaded)

    # Raises ValueError if not a well-formed UUID hex string.
    assert uuid.UUID(data_file["file_id"]).hex == data_file["file_id"]


def test_register_uploaded_file_id_is_not_derived_from_filename():
    first = register_uploaded_file(FakeUploadedFile("report.xlsx", _excel_bytes(100.0)))
    second = register_uploaded_file(FakeUploadedFile("report.xlsx", _excel_bytes(200.0)))

    assert first["file_id"] != second["file_id"]
    assert first["filename"] == second["filename"] == "report.xlsx"


def test_register_uploaded_file_preserves_original_filename():
    uploaded = FakeUploadedFile("Q1 Sales (final).xlsx", _excel_bytes())

    data_file = register_uploaded_file(uploaded)

    assert data_file["filename"] == "Q1 Sales (final).xlsx"


def test_register_uploaded_file_defaults_display_name_to_filename():
    uploaded = FakeUploadedFile("Q1 Sales (final).xlsx", _excel_bytes())

    data_file = register_uploaded_file(uploaded)

    assert data_file["display_name"] == "Q1 Sales (final).xlsx"


def test_register_uploaded_file_defaults_role_to_none():
    uploaded = FakeUploadedFile("report.xlsx", _excel_bytes())

    data_file = register_uploaded_file(uploaded)

    assert data_file["role"] is None


def test_register_uploaded_file_performs_no_analysis():
    uploaded = FakeUploadedFile("report.xlsx", _excel_bytes())

    data_file = register_uploaded_file(uploaded)

    assert data_file["analysis"] is None
    assert data_file["value_column"] is None
    assert data_file["period_column"] is None
    assert "numeric_summary" not in data_file
    assert "profile" not in data_file


def test_register_uploaded_file_does_not_mutate_uploaded_file_object():
    uploaded = FakeUploadedFile("report.xlsx", _excel_bytes())
    content_before = uploaded.getbuffer()

    register_uploaded_file(uploaded)

    assert uploaded.getbuffer() == content_before
    assert uploaded.name == "report.xlsx"


# ==================================================
# determine_active_file_id — unit tests
# ==================================================


def test_determine_active_file_id_with_no_files_is_none():
    assert determine_active_file_id({}) is None


def test_determine_active_file_id_with_exactly_one_file():
    data_files = {"abc123": {"file_id": "abc123"}}

    assert determine_active_file_id(data_files) == "abc123"


def test_determine_active_file_id_with_multiple_files_is_none():
    data_files = {"abc123": {"file_id": "abc123"}, "def456": {"file_id": "def456"}}

    assert determine_active_file_id(data_files) is None


# ==================================================
# Streamlit AppTest — multi-file upload wiring
# ==================================================


def _upload(at, files):
    at.run()
    at.file_uploader[0].set_value(files)
    at.run()
    return at


def test_single_file_upload_registers_one_data_file_and_sets_active():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at = _upload(at, ("report.xlsx", _excel_bytes(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"))

    assert not at.exception
    data_files = at.session_state["data_files"]
    assert len(data_files) == 1
    [file_id] = data_files.keys()
    assert at.session_state["active_file_id"] == file_id
    assert data_files[file_id]["filename"] == "report.xlsx"
    # V0.7.2: the active file's analysis is now actually run and stored here
    # (V0.7.1 left it None since analyze_file() didn't exist yet).
    assert data_files[file_id]["analysis"] is not None
    assert data_files[file_id]["analysis"]["profile"]["row_count"] == 1


def test_multiple_file_upload_registers_independent_identities():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    at = _upload(at, [
        ("january.xlsx", _excel_bytes(100.0), mime),
        ("february.xlsx", _excel_bytes(200.0), mime),
    ])

    assert not at.exception
    data_files = at.session_state["data_files"]
    assert len(data_files) == 2
    assert len({data_files[fid]["file_id"] for fid in data_files}) == 2
    assert at.session_state["active_file_id"] is None


def test_duplicate_filenames_produce_different_file_ids_via_apptest():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    at = _upload(at, [
        ("data.xlsx", _excel_bytes(100.0), mime),
        ("data.xlsx", _excel_bytes(200.0), mime),
    ])

    assert not at.exception
    data_files = at.session_state["data_files"]
    assert len(data_files) == 2
    filenames = [data_files[fid]["filename"] for fid in data_files]
    assert filenames == ["data.xlsx", "data.xlsx"]
    assert len(set(data_files.keys())) == 2


def test_rerun_does_not_duplicate_already_registered_file():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at = _upload(at, ("report.xlsx", _excel_bytes(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"))
    assert len(at.session_state["data_files"]) == 1

    at.run()  # simulate a rerun triggered by an unrelated widget interaction
    at.run()

    assert not at.exception
    assert len(at.session_state["data_files"]) == 1


def test_no_upload_leaves_data_files_empty():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    assert not at.exception
    assert at.session_state["data_files"] == {}
    assert at.session_state["active_file_id"] is None


# ==================================================
# Streamlit AppTest - roles, remove/clear, comparison Q&A/chart/insight
# (V0.7.5-V0.7.8)
# ==================================================


def _revenue_excel_bytes(value=100.0):
    df = pd.DataFrame({
        "period": pd.to_datetime(["2024-01-01", "2024-02-01"]),
        "revenue": [value, value + 10.0],
    })
    buffer = BytesIO()
    df.to_excel(buffer, index=False, engine="openpyxl")
    return buffer.getvalue()


def _upload_two_files(at):
    mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    at.run()
    at.file_uploader[0].set_value([
        ("january.xlsx", _revenue_excel_bytes(100.0), mime),
        ("february.xlsx", _revenue_excel_bytes(200.0), mime),
    ])
    at.run()
    return at


def _assign_roles(at, roles: dict):
    for file_id, role in roles.items():
        at.selectbox(key=f"role_select_{file_id}").set_value(role)
    at.run()
    return at


@pytest.fixture
def _gemini_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")


def test_two_files_render_role_assignment_ui():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at = _upload_two_files(at)

    assert not at.exception
    subheaders = [el.value for el in at.subheader]
    assert "Uploaded Files" in subheaders
    file_ids = list(at.session_state["data_files"].keys())
    assert any(w.key == f"role_select_{file_ids[0]}" for w in at.selectbox)
    assert any(w.key == f"remove_{file_ids[0]}" for w in at.button)


def test_assigning_roles_updates_data_file_role():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at = _upload_two_files(at)
    file_ids = list(at.session_state["data_files"].keys())

    at = _assign_roles(at, {file_ids[0]: "previous", file_ids[1]: "current"})

    assert not at.exception
    assert at.session_state["data_files"][file_ids[0]]["role"] == "previous"
    assert at.session_state["data_files"][file_ids[1]]["role"] == "current"


def test_removing_a_file_drops_it_and_reruns_to_single_file_view():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at = _upload_two_files(at)
    file_ids = list(at.session_state["data_files"].keys())

    at.button(key=f"remove_{file_ids[0]}").click()
    at.run()

    assert not at.exception
    assert file_ids[0] not in at.session_state["data_files"]
    assert len(at.session_state["data_files"]) == 1


def test_clear_all_files_empties_everything():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at = _upload_two_files(at)

    at.button(key="clear_all_files").click()
    at.run()

    assert not at.exception
    assert at.session_state["data_files"] == {}
    assert at.session_state["qa_answers"] == {}
    assert at.session_state["chart_specs"] == {}
    assert at.session_state["ai_results"] == {}
    assert at.session_state["comparison_result"] is None


def test_comparison_section_shows_missing_api_key_message_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at = _upload_two_files(at)
    file_ids = list(at.session_state["data_files"].keys())
    at = _assign_roles(at, {file_ids[0]: "previous", file_ids[1]: "current"})

    assert not at.exception
    infos = [el.value for el in at.info]
    assert any("GEMINI_API_KEY" in text for text in infos)


def test_valid_comparison_question_produces_deterministic_answer(monkeypatch, _gemini_api_key):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at = _upload_two_files(at)
    file_ids = list(at.session_state["data_files"].keys())
    at = _assign_roles(at, {file_ids[0]: "previous", file_ids[1]: "current"})

    intent_response = json.dumps({
        "intent": "file_value_comparison", "column_hint": "revenue", "metric": "sum",
        "period_hint": None, "from_file_hint": "previous", "to_file_hint": "current",
    })
    monkeypatch.setattr(GeminiProvider, "generate", lambda self, prompt: intent_response)
    at.text_input(key="comparison_question").set_value("How does revenue compare?")
    at.run()
    at.button(key="comparison_get_answer").click()
    at.run()

    assert not at.exception
    answer_texts = [el.value for el in at.markdown]
    # january.xlsx revenue = [100, 110] -> sum 210.0 (previous)
    # february.xlsx revenue = [200, 210] -> sum 410.0 (current)
    assert any("210.0" in text and "410.0" in text for text in answer_texts)
    assert at.session_state["comparison_result"]["found"] is True


def test_unsupported_comparison_question_produces_deterministic_message(monkeypatch, _gemini_api_key):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at = _upload_two_files(at)
    file_ids = list(at.session_state["data_files"].keys())
    at = _assign_roles(at, {file_ids[0]: "previous", file_ids[1]: "current"})

    intent_response = json.dumps({
        "intent": "unsupported", "column_hint": None, "metric": None,
        "period_hint": None, "from_file_hint": None, "to_file_hint": None,
    })
    monkeypatch.setattr(GeminiProvider, "generate", lambda self, prompt: intent_response)
    at.text_input(key="comparison_question").set_value("What is the weather today?")
    at.run()
    at.button(key="comparison_get_answer").click()
    at.run()

    assert not at.exception
    assert at.session_state["comparison_result"] is None


def test_comparison_missing_role_produces_deterministic_error_message(monkeypatch, _gemini_api_key):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at = _upload_two_files(at)
    # Deliberately do not assign roles.

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
    assert at.session_state["comparison_result"] is None


def test_valid_comparison_chart_request_renders_chart(monkeypatch, _gemini_api_key):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at = _upload_two_files(at)
    file_ids = list(at.session_state["data_files"].keys())
    at = _assign_roles(at, {file_ids[0]: "previous", file_ids[1]: "current"})

    chart_intent_response = json.dumps({
        "intent": "file_comparison_chart", "column_hint": "revenue", "metric": "sum",
        "period_hint": None, "from_file_hint": "previous", "to_file_hint": "current",
    })
    monkeypatch.setattr(GeminiProvider, "generate", lambda self, prompt: chart_intent_response)
    at.text_input(key="comparison_chart_question").set_value("Chart revenue previous vs current")
    at.run()
    at.button(key="comparison_show_chart").click()
    at.run()

    assert not at.exception
    spec = at.session_state["comparison_chart_spec"]
    assert spec is not None and spec["found"] is True
    assert at.session_state["comparison_result"]["found"] is True


def test_comparison_ai_insight_generates_after_valid_question(monkeypatch, _gemini_api_key):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at = _upload_two_files(at)
    file_ids = list(at.session_state["data_files"].keys())
    at = _assign_roles(at, {file_ids[0]: "previous", file_ids[1]: "current"})

    intent_response = json.dumps({
        "intent": "file_value_comparison", "column_hint": "revenue", "metric": "sum",
        "period_hint": None, "from_file_hint": "previous", "to_file_hint": "current",
    })
    monkeypatch.setattr(GeminiProvider, "generate", lambda self, prompt: intent_response)
    at.text_input(key="comparison_question").set_value("How does revenue compare?")
    at.run()
    at.button(key="comparison_get_answer").click()
    at.run()

    insight_response = json.dumps({
        "summary": "Revenue increased between the two files.",
        "key_insights": ["Growth was observed."],
        "trend_interpretation": None,
        "warnings": [],
        "recommendations": [],
    })
    monkeypatch.setattr(GeminiProvider, "generate", lambda self, prompt: insight_response)
    at.button(key="comparison_ai_insight_button").click()
    at.run()

    assert not at.exception
    markdown_texts = [el.value for el in at.markdown]
    assert any("Revenue increased between the two files." in text for text in markdown_texts)


def test_removing_a_file_clears_stale_comparison_result(monkeypatch, _gemini_api_key):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at = _upload_two_files(at)
    file_ids = list(at.session_state["data_files"].keys())
    at = _assign_roles(at, {file_ids[0]: "previous", file_ids[1]: "current"})

    intent_response = json.dumps({
        "intent": "file_value_comparison", "column_hint": "revenue", "metric": "sum",
        "period_hint": None, "from_file_hint": "previous", "to_file_hint": "current",
    })
    monkeypatch.setattr(GeminiProvider, "generate", lambda self, prompt: intent_response)
    at.text_input(key="comparison_question").set_value("Compare revenue")
    at.run()
    at.button(key="comparison_get_answer").click()
    at.run()
    assert at.session_state["comparison_result"]["found"] is True

    at.button(key=f"remove_{file_ids[0]}").click()
    at.run()

    assert not at.exception
    assert at.session_state["comparison_result"] is None
    assert at.session_state["comparison_qa_answer"] is None


def test_active_file_qa_results_are_isolated_per_file(monkeypatch, _gemini_api_key):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    at.run()
    at.file_uploader[0].set_value(("first.xlsx", _revenue_excel_bytes(100.0), mime))
    at.run()
    first_file_id = next(iter(at.session_state["data_files"]))

    intent_response = json.dumps({
        "intent": "column_stat", "metric": "sum", "column_hint": "revenue",
        "period_hint": None, "from_period_hint": None, "to_period_hint": None,
    })
    monkeypatch.setattr(GeminiProvider, "generate", lambda self, prompt: intent_response)
    at.text_input(key="qa_question").set_value("What is the total revenue?")
    at.run()
    at.button(key="qa_get_answer").click()
    at.run()
    assert at.session_state["qa_answers"][first_file_id] is not None

    at.button(key="clear_all_files").click()
    at.run()
    at.file_uploader[0].set_value(("second.xlsx", _revenue_excel_bytes(500.0), mime))
    at.run()
    second_file_id = next(iter(at.session_state["data_files"]))

    assert second_file_id != first_file_id
    assert at.session_state["qa_answers"] == {}
