import json
import os
import tempfile
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from src.ai_interpreter import interpret
from src.chart_builder import (
    build_missing_values_chart,
    build_numeric_summary_chart,
    build_period_change_chart,
    build_trend_chart,
)
from src.chart_intent_interpreter import interpret_chart_question
from src.comparison_ai_payload import build_comparison_ai_payload
from src.comparison_prompt_builder import build_comparison_prompt
from src.excel_loader import load_excel
from src.file_analysis import (
    analyze_file,
    date_like_column_names,
    format_column_option,
    numeric_column_names,
)
from src.gemini_provider import GeminiProvider
from src.multi_file_chart_intent_interpreter import interpret_multi_file_chart_question
from src.multi_file_comparison import dispatch_comparison_intent
from src.multi_file_qa_interpreter import interpret_multi_file_question
from src.qa_answer import answer_comparison_result, answer_grounded_result
from src.qa_chart_engine import build_chart_spec, build_multi_file_chart_spec
from src.qa_chart_renderer import render_chart_spec
from src.qa_engine import dispatch_intent
from src.qa_interpreter import interpret_question

ROLE_OPTIONS = [None, "previous", "current", "reference", "comparison"]

AI_ERROR_MESSAGES = {
    "missing_api_key": "GEMINI_API_KEY is not set. Add it to your environment to enable AI insights.",
    "provider_unavailable": "The Gemini SDK is not available in this environment.",
    "provider_error": "The Gemini API request failed. Please try again.",
    "empty_response": "Gemini returned an empty response.",
    "malformed_output": "Gemini returned a response that could not be parsed.",
}


def save_uploaded_file(uploaded_file, directory: Path) -> Path:
    """Persist an uploaded Streamlit file to `directory`, preserving its extension."""
    suffix = Path(uploaded_file.name).suffix
    dest = Path(directory) / f"upload{suffix}"
    dest.write_bytes(uploaded_file.getbuffer())
    return dest


def register_uploaded_file(uploaded_file) -> dict:
    """Load an uploaded Streamlit file into a new DataFile record (V0.7.1).

    Reuses the existing save-to-temp-dir-then-load mechanism unchanged.
    Assigns a fresh, filename-independent identity (`uuid.uuid4().hex`) so
    that two uploads sharing the same filename never collide. Performs no
    analysis — that is V0.7.2 (`analysis` is always None here).

    `_source_upload_id` is an internal bookkeeping field (not part of the
    public DataFile shape) recording Streamlit's own per-upload identity
    (`uploaded_file.file_id`, when present), so that `main()` can avoid
    re-registering the same widget upload as a new DataFile on every rerun.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        excel_path = save_uploaded_file(uploaded_file, Path(tmp_dir))
        raw_df = load_excel(excel_path)

    return {
        "file_id": uuid.uuid4().hex,
        "filename": uploaded_file.name,
        "display_name": uploaded_file.name,
        "role": None,
        "raw_df": raw_df,
        "value_column": None,
        "period_column": None,
        "analysis": None,
        "_source_upload_id": getattr(uploaded_file, "file_id", None),
    }


def determine_active_file_id(data_files: dict) -> str | None:
    """Return the file_id that should be active: the sole file's id when
    exactly one is registered, otherwise None. Multi-file active-selection
    is not yet implemented (V0.7.2+) — never guess among multiple files."""
    if len(data_files) == 1:
        return next(iter(data_files))
    return None


PER_FILE_STATE_KEYS = ("qa_answers", "chart_specs", "ai_results", "ai_cache_keys")
COMPARISON_STATE_KEYS = (
    "comparison_qa_answer", "comparison_chart_spec",
    "comparison_ai_result", "comparison_ai_cache_key", "comparison_result",
)


def init_session_state() -> None:
    """Ensure every V0.7 session_state container exists exactly once per
    session. Per-file results (`qa_answers`, `chart_specs`, `ai_results`,
    `ai_cache_keys`) are dicts keyed by file_id, not bare values — this is
    what prevents a stale result from one file silently appearing to
    belong to a different file once `active_file_id` changes (e.g. after a
    file is removed)."""
    if "data_files" not in st.session_state:
        st.session_state["data_files"] = {}
    if "seen_upload_ids" not in st.session_state:
        st.session_state["seen_upload_ids"] = set()
    for key in PER_FILE_STATE_KEYS:
        if key not in st.session_state:
            st.session_state[key] = {}
    for key in COMPARISON_STATE_KEYS:
        if key not in st.session_state:
            st.session_state[key] = None


def forget_file(file_id: str) -> None:
    """Remove a file and every result keyed by its file_id. Also clears the
    comparison state, since any stored comparison may have involved the
    removed file and would otherwise silently look valid for a different
    pair of files."""
    st.session_state["data_files"].pop(file_id, None)
    for key in PER_FILE_STATE_KEYS:
        st.session_state[key].pop(file_id, None)
    for key in COMPARISON_STATE_KEYS:
        st.session_state[key] = None


def clear_all_files() -> None:
    """Remove every uploaded file and every per-file/comparison result."""
    st.session_state["data_files"] = {}
    for key in PER_FILE_STATE_KEYS:
        st.session_state[key] = {}
    for key in COMPARISON_STATE_KEYS:
        st.session_state[key] = None


def build_analytics_payload(numeric_summary: dict, date_summary: dict, trend: dict | None = None,
                             period_comparison: dict | None = None, anomalies: dict | None = None) -> dict:
    payload = {"numeric_summary": numeric_summary, "date_summary": date_summary}
    if trend is not None:
        payload["trend"] = trend
    if period_comparison is not None:
        payload["period_comparison"] = period_comparison
    if anomalies is not None:
        payload["anomalies"] = anomalies
    return payload


def ai_error_message(reason: str) -> str:
    return AI_ERROR_MESSAGES.get(reason, "AI insight generation failed.")


CHART_ERROR_MESSAGES = {
    "invalid_json": "Could not understand the chart request.",
    "invalid_schema": "Could not understand the chart request.",
    "missing_api_key": "GEMINI_API_KEY is not set. Add it to your environment to enable chart requests.",
    "provider_unavailable": "The Gemini SDK is not available in this environment.",
    "provider_error": "The Gemini API request failed. Please try again.",
    "column_not_found": "The requested column was not found.",
    "ambiguous_column": "The requested column reference is ambiguous. Please be more specific.",
    "column_required": "Please specify which column to chart.",
    "column_series_mismatch": "The requested column does not match the currently selected trend series.",
    "period_not_found": "The requested period was not found.",
    "ambiguous_period": "The requested period reference is ambiguous. Please be more specific.",
    "period_range_incomplete": "Please specify both a starting and an ending period.",
    "period_change_not_found": "No change was found between the requested periods.",
    "unsupported_metric": "That statistic isn't supported for charting.",
    "unsupported": "That chart request isn't supported yet.",
    "file_not_found": "I couldn't find a file matching that reference.",
    "insufficient_files": "At least two distinct files are needed for a comparison chart.",
    "role_not_assigned": "No file has been assigned that role yet.",
    "duplicate_role": "That reference matches more than one file — please be more specific.",
    "column_not_found_in_file_a": "I couldn't find that column in the first file.",
    "column_not_found_in_file_b": "I couldn't find that column in the second file.",
    "ambiguous_column_in_file_a": "That column reference is ambiguous in the first file.",
    "ambiguous_column_in_file_b": "That column reference is ambiguous in the second file.",
    "unsupported_comparison": "That comparison chart isn't supported yet.",
}


def chart_error_message(reason: str) -> str:
    return CHART_ERROR_MESSAGES.get(reason, "The requested chart cannot be generated from the available analysis.")


def build_qa_analysis_payload(numeric_summary: dict, profile: dict, period_comparison: dict | None = None,
                               anomalies: dict | None = None) -> dict:
    payload = {"numeric_summary": numeric_summary, "profile": profile}
    if period_comparison is not None:
        payload["period_comparison"] = period_comparison
    if anomalies is not None:
        payload["anomalies"] = anomalies
    return payload


def known_file_labels(data_files: dict) -> list[str]:
    """Roles and display names of every registered file, for LLM context
    only (never the internal file_id) — mirrors how column names are
    already passed as context without being resolved by the LLM."""
    labels = set()
    for data_file in data_files.values():
        if data_file.get("role"):
            labels.add(data_file["role"])
        if data_file.get("display_name"):
            labels.add(data_file["display_name"])
    return sorted(labels)


def known_numeric_column_names(data_files: dict) -> list[str]:
    names = set()
    for data_file in data_files.values():
        analysis = data_file.get("analysis")
        if analysis:
            names.update(c["name"] for c in analysis["numeric_summary"]["columns"])
    return sorted(names)


def render_ai_insight_result(result: dict | None) -> None:
    """Shared rendering for both the single-file and comparison AI Insight
    sections — identical presentation, just called with a different
    already-computed `interpret()` result."""
    if result is None:
        return
    if result["is_valid"]:
        insight = result["insight"]
        st.write(insight["summary"])
        if insight["key_insights"]:
            st.write("**Key insights**")
            for item in insight["key_insights"]:
                st.write(f"- {item}")
        if insight["trend_interpretation"]:
            st.write(f"**Interpretation:** {insight['trend_interpretation']}")
        if insight["warnings"]:
            st.write("**Warnings**")
            for item in insight["warnings"]:
                st.write(f"- {item}")
        if insight["recommendations"]:
            st.write("**Recommendations**")
            for item in insight["recommendations"]:
                st.write(f"- {item}")
    else:
        st.error(ai_error_message(result["reason"]))


def render_comparison_section(data_files: dict) -> None:
    """Multi-file Q&A / chart / AI insight (V0.7.5-V0.7.6). Only reachable
    once 2+ files are registered. Reuses the existing deterministic
    multi_file_comparison engine and the existing AIProvider/interpret()
    machinery — the LLM only classifies the request; every number shown
    here is read verbatim from dispatch_comparison_intent()'s output.
    """
    st.subheader("Compare Files")
    st.caption(
        "Ask a question or request a chart comparing two files by role "
        "(e.g. \"previous\"/\"current\") or display name. The AI only "
        "classifies your request — the comparison itself is always computed "
        "deterministically from the selected files' already-computed analysis."
    )

    if not os.environ.get("GEMINI_API_KEY"):
        st.info(ai_error_message("missing_api_key"))
        return

    column_names = known_numeric_column_names(data_files)
    file_labels = known_file_labels(data_files)

    question = st.text_input("Ask a question comparing two files", key="comparison_question")
    if st.button("Get Comparison Answer", key="comparison_get_answer"):
        if not question.strip():
            st.session_state["comparison_qa_answer"] = None
        else:
            provider = GeminiProvider()
            interpretation = interpret_multi_file_question(question, column_names, file_labels, provider)
            if interpretation["is_valid"]:
                comparison_result = dispatch_comparison_intent(interpretation["intent"], data_files)
                st.session_state["comparison_qa_answer"] = answer_comparison_result(comparison_result)
                st.session_state["comparison_result"] = comparison_result if comparison_result["found"] else None
            else:
                st.session_state["comparison_qa_answer"] = answer_comparison_result(interpretation)
                st.session_state["comparison_result"] = None

    if st.session_state.get("comparison_qa_answer"):
        st.write(st.session_state["comparison_qa_answer"])

    chart_question = st.text_input("Describe a chart comparing two files", key="comparison_chart_question")
    if st.button("Show Comparison Chart", key="comparison_show_chart"):
        if not chart_question.strip():
            st.session_state["comparison_chart_spec"] = None
        else:
            provider = GeminiProvider()
            chart_interpretation = interpret_multi_file_chart_question(chart_question, column_names, file_labels, provider)
            if chart_interpretation["is_valid"]:
                chart_spec = build_multi_file_chart_spec(chart_interpretation["intent"], data_files)
                st.session_state["comparison_chart_spec"] = chart_spec
                if chart_spec["found"]:
                    value_intent = {
                        "intent": "file_value_comparison",
                        "column_hint": chart_interpretation["intent"].get("column_hint"),
                        "metric": chart_interpretation["intent"].get("metric"),
                        "period_hint": None,
                        "from_file_hint": chart_interpretation["intent"].get("from_file_hint"),
                        "to_file_hint": chart_interpretation["intent"].get("to_file_hint"),
                    }
                    st.session_state["comparison_result"] = dispatch_comparison_intent(value_intent, data_files)
            else:
                st.session_state["comparison_chart_spec"] = {"found": False, "reason": chart_interpretation["reason"]}

    comparison_chart_spec = st.session_state.get("comparison_chart_spec")
    if comparison_chart_spec is not None:
        if comparison_chart_spec["found"]:
            figure = render_chart_spec(comparison_chart_spec)
            if figure is not None:
                st.plotly_chart(figure, use_container_width=True, key="comparison_chart_result")
        else:
            st.error(chart_error_message(comparison_chart_spec["reason"]))

    st.write("**AI Comparison Insight**")
    comparison_result = st.session_state.get("comparison_result")
    if comparison_result is None:
        st.caption("Ask a comparison question or request a comparison chart above to enable this.")
        return

    ai_payload = build_comparison_ai_payload(comparison_result)
    cache_key = json.dumps(ai_payload, sort_keys=True, default=str)

    if st.button("Generate Comparison Insight", key="comparison_ai_insight_button"):
        provider = GeminiProvider()
        st.session_state["comparison_ai_result"] = interpret(ai_payload, provider, prompt_builder=build_comparison_prompt)
        st.session_state["comparison_ai_cache_key"] = cache_key

    if st.session_state.get("comparison_ai_cache_key") == cache_key:
        render_ai_insight_result(st.session_state.get("comparison_ai_result"))


def main():
    st.title("AI Excel Analyst")
    st.write(
        "Upload an Excel file to generate deterministic analytics and an "
        "AI-generated summary of the results. Calculations are always "
        "produced by the analytics engine — the AI only interprets them."
    )

    uploaded_files = st.file_uploader("Upload Excel file(s)", type=["xlsx", "xls"], accept_multiple_files=True)

    init_session_state()

    # Tracked across the whole session (never cleared by forget_file() or
    # clear_all_files()) because the file_uploader widget itself keeps
    # holding a removed/cleared file's bytes until the user changes the
    # widget selection — without this, "Remove"/"Clear All Files" would
    # only clear app-level state for one rerun before the same upload got
    # silently re-registered as a brand-new DataFile on the very next run.
    seen_upload_ids = st.session_state["seen_upload_ids"]
    for uploaded in uploaded_files or []:
        upload_id = getattr(uploaded, "file_id", None)
        if upload_id is not None and upload_id in seen_upload_ids:
            continue
        try:
            data_file = register_uploaded_file(uploaded)
        except Exception as exc:
            st.error(f"Could not read the uploaded file: {exc}")
            continue
        st.session_state["data_files"][data_file["file_id"]] = data_file
        if upload_id is not None:
            seen_upload_ids.add(upload_id)

    st.session_state["active_file_id"] = determine_active_file_id(st.session_state["data_files"])
    active_file_id = st.session_state["active_file_id"]
    data_files = st.session_state["data_files"]

    if not data_files:
        return

    if st.button("Clear All Files", key="clear_all_files"):
        clear_all_files()
        st.rerun()

    if len(data_files) >= 2:
        st.subheader("Uploaded Files")
        st.caption("Assign roles (e.g. previous/current) to enable comparisons across files.")
        for file_id, data_file in list(data_files.items()):
            name_col, role_col, remove_col = st.columns([3, 2, 1])
            name_col.write(data_file["display_name"])
            current_role = data_file.get("role")
            role_index = ROLE_OPTIONS.index(current_role) if current_role in ROLE_OPTIONS else 0
            selected_role = role_col.selectbox(
                "Role", ROLE_OPTIONS, index=role_index,
                format_func=lambda r: r or "(none)", key=f"role_select_{file_id}",
            )
            data_file["role"] = selected_role
            # Pre-analyze each file's numeric columns so the comparison
            # section below has column context, reusing analyze_file()
            # rather than inspecting raw_df directly from app.py.
            if data_file.get("analysis") is None:
                data_file["analysis"] = analyze_file(data_file["raw_df"])
            if remove_col.button("Remove", key=f"remove_{file_id}"):
                forget_file(file_id)
                st.rerun()

        render_comparison_section(data_files)
        return

    data_file = data_files[active_file_id]
    raw_df = data_file["raw_df"]

    analysis = analyze_file(raw_df)
    profile = analysis["profile"]

    st.subheader("Dataset Information")
    st.write(f"Rows: {profile['row_count']} | Columns: {profile['column_count']}")
    if profile["duplicate_row_count"]:
        st.warning(f"{profile['duplicate_row_count']} duplicate row(s) detected.")

    if profile["is_empty"]:
        st.error("The uploaded file contains no rows.")
        return

    st.subheader("Column Information")
    st.dataframe(pd.DataFrame(profile["columns"]))
    st.plotly_chart(build_missing_values_chart(profile), use_container_width=True)

    normalized_df = analysis["normalized_df"]
    numeric_summary = analysis["numeric_summary"]
    date_summary = analysis["date_summary"]

    st.subheader("Deterministic Analytics")
    st.write("Numeric columns")
    if numeric_summary["columns"]:
        st.dataframe(pd.DataFrame(numeric_summary["columns"]))
        st.plotly_chart(
            build_numeric_summary_chart(numeric_summary),
            use_container_width=True,
        )
    else:
        st.info("No numeric columns found.")

    st.write("Date columns")
    if date_summary["columns"]:
        st.dataframe(pd.DataFrame(date_summary["columns"]))
    else:
        st.info("No usable date columns found.")

    trend = None
    period_comparison = None
    anomalies = None
    monthly_series = None
    value_options = numeric_column_names(normalized_df)
    period_options = date_like_column_names(profile)

    st.subheader("Trend Analysis")
    if not value_options and not period_options:
        st.info("No numeric columns and no usable date columns found; trend analysis is unavailable.")
    elif not value_options:
        st.info("No numeric columns found; trend analysis is unavailable.")
    elif not period_options:
        st.info("No usable date columns found; trend analysis is unavailable.")
    else:
        value_option = st.selectbox(
            "Value column", value_options, format_func=lambda opt: format_column_option(opt, value_options)
        )
        period_option = st.selectbox(
            "Period column", period_options, format_func=lambda opt: format_column_option(opt, period_options)
        )
        st.caption("Values are aggregated by month (sum) before trend analysis.")

        data_file["value_column"] = value_option
        data_file["period_column"] = period_option

        analysis = analyze_file(raw_df, value_option[0], period_option[0])
        monthly_series = analysis["monthly_series"]
        trend = analysis["trend"]
        period_comparison = analysis["period_comparison"]
        anomalies = analysis["anomalies"]

        if monthly_series is None:
            st.error(
                "Could not aggregate the selected period column: "
                "Period column does not contain datetime-compatible values."
            )
        else:
            st.write(f"Trend: {trend['trend']}")
            st.plotly_chart(
                build_trend_chart(monthly_series, anomalies=anomalies),
                use_container_width=True,
            )
            if period_comparison["comparisons"]:
                st.dataframe(pd.DataFrame(period_comparison["comparisons"]))
                st.plotly_chart(
                    build_period_change_chart(period_comparison),
                    use_container_width=True,
                )

    data_file["analysis"] = analysis

    analytics_payload = build_analytics_payload(numeric_summary, date_summary, trend, period_comparison, anomalies)

    st.subheader("AI-Generated Insight")

    if not os.environ.get("GEMINI_API_KEY"):
        st.error(ai_error_message("missing_api_key"))
        return

    cache_key = json.dumps(analytics_payload, sort_keys=True, default=str)

    if st.button("Generate AI Insight"):
        provider = GeminiProvider()
        st.session_state["ai_results"][active_file_id] = interpret(analytics_payload, provider)
        st.session_state["ai_cache_keys"][active_file_id] = cache_key

    if st.session_state["ai_cache_keys"].get(active_file_id) == cache_key:
        render_ai_insight_result(st.session_state["ai_results"].get(active_file_id))

    st.subheader("Ask a Question")
    st.caption(
        "The AI only classifies your question and extracts hints from it; the "
        "answer itself is looked up and computed deterministically from the "
        "analytics above — the AI never generates the numeric answer."
    )
    question = st.text_input("Ask a question about this dataset", key="qa_question")
    if st.button("Get Answer", key="qa_get_answer"):
        if not question.strip():
            st.session_state["qa_answers"][active_file_id] = None
        else:
            provider = GeminiProvider()
            column_names = [c["name"] for c in profile["columns"]]
            interpretation = interpret_question(question, column_names, provider)
            if interpretation["is_valid"]:
                qa_payload = build_qa_analysis_payload(numeric_summary, profile, period_comparison, anomalies)
                grounded_result = dispatch_intent(interpretation["intent"], qa_payload, monthly_series)
                st.session_state["qa_answers"][active_file_id] = answer_grounded_result(grounded_result)
            else:
                st.session_state["qa_answers"][active_file_id] = answer_grounded_result(interpretation)

    if st.session_state["qa_answers"].get(active_file_id):
        st.write(st.session_state["qa_answers"][active_file_id])

    st.subheader("Ask for a Chart")
    st.caption(
        "The AI only classifies your chart request; the chart itself is always "
        "built from the deterministic analytics above — the AI never generates "
        "chart data or Plotly code."
    )
    chart_question = st.text_input("Describe the chart you want", key="chart_question")
    if st.button("Show Chart", key="chart_show_chart"):
        if not chart_question.strip():
            st.session_state["chart_specs"][active_file_id] = None
        else:
            provider = GeminiProvider()
            column_names = [c["name"] for c in profile["columns"]]
            chart_interpretation = interpret_chart_question(chart_question, column_names, provider)
            if chart_interpretation["is_valid"]:
                chart_payload = build_qa_analysis_payload(numeric_summary, profile, period_comparison, anomalies)
                st.session_state["chart_specs"][active_file_id] = build_chart_spec(
                    chart_interpretation["intent"], chart_payload, monthly_series, anomalies,
                )
            else:
                st.session_state["chart_specs"][active_file_id] = {"found": False, "reason": chart_interpretation["reason"]}

    chart_spec = st.session_state["chart_specs"].get(active_file_id)
    if chart_spec is not None:
        if chart_spec["found"]:
            figure = render_chart_spec(chart_spec)
            if figure is not None:
                st.plotly_chart(figure, use_container_width=True, key="chart_question_result")
        else:
            st.error(chart_error_message(chart_spec["reason"]))


if __name__ == "__main__":
    main()
