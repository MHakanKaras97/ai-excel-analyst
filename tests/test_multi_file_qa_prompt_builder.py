from src.multi_file_qa_prompt_builder import build_multi_file_qa_prompt

SUPPORTED_INTENTS = (
    "file_value_comparison",
    "file_period_comparison",
    "file_trend_comparison",
    "file_anomaly_comparison",
    "unsupported",
)


def test_build_multi_file_qa_prompt_returns_a_string():
    prompt = build_multi_file_qa_prompt("How does revenue compare?", ["Revenue"], ["previous", "current"])

    assert isinstance(prompt, str) and len(prompt) > 0


def test_build_multi_file_qa_prompt_includes_question_columns_and_files():
    prompt = build_multi_file_qa_prompt("Compare totals", ["Revenue", "Region"], ["previous", "current"])

    assert "Compare totals" in prompt
    assert "Revenue" in prompt and "Region" in prompt
    assert "previous" in prompt and "current" in prompt


def test_build_multi_file_qa_prompt_represents_all_five_intents():
    prompt = build_multi_file_qa_prompt("q", [], [])

    for intent in SUPPORTED_INTENTS:
        assert intent in prompt


def test_build_multi_file_qa_prompt_documents_from_to_file_hints():
    prompt = build_multi_file_qa_prompt("q", [], [])

    assert "from_file_hint" in prompt
    assert "to_file_hint" in prompt


def test_build_multi_file_qa_prompt_prohibits_calculation_and_resolution():
    prompt = build_multi_file_qa_prompt("q", [], [])

    assert "calculate anything" in prompt
    assert "NEVER invent a value for any hint" in prompt
    assert "resolution" in prompt.lower()


def test_build_multi_file_qa_prompt_instructs_json_only_output():
    prompt = build_multi_file_qa_prompt("q", [], [])

    assert "Return JSON only" in prompt


def test_build_multi_file_qa_prompt_handles_empty_lists_without_error():
    prompt = build_multi_file_qa_prompt("q", [], [])

    assert isinstance(prompt, str)


def test_build_multi_file_qa_prompt_is_pure_no_side_effects():
    columns, files = ["Revenue"], ["previous"]
    before_columns, before_files = list(columns), list(files)

    build_multi_file_qa_prompt("q", columns, files)

    assert columns == before_columns and files == before_files
