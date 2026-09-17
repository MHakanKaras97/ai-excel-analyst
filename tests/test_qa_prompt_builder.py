from src.qa_prompt_builder import build_qa_prompt

SUPPORTED_INTENTS = (
    "period_value",
    "period_extremum",
    "period_change",
    "column_stat",
    "anomaly_check",
    "missing_values",
    "unsupported",
)

SUPPORTED_METRICS = ("sum", "mean", "median", "min", "max", "count")


def test_build_qa_prompt_returns_a_string():
    prompt = build_qa_prompt("What was the value in March?", ["Revenue", "Month"])

    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_build_qa_prompt_includes_the_user_question():
    prompt = build_qa_prompt("What was the peak month for Revenue?", ["Revenue", "Month"])

    assert "What was the peak month for Revenue?" in prompt


def test_build_qa_prompt_includes_supplied_column_names():
    prompt = build_qa_prompt("How many missing values?", ["Revenue", "Region", "Date"])

    assert "Revenue" in prompt
    assert "Region" in prompt
    assert "Date" in prompt


def test_build_qa_prompt_represents_all_seven_intents():
    prompt = build_qa_prompt("irrelevant question", [])

    for intent in SUPPORTED_INTENTS:
        assert intent in prompt


def test_build_qa_prompt_represents_metric_enum():
    prompt = build_qa_prompt("irrelevant question", [])

    for metric in SUPPORTED_METRICS:
        assert metric in prompt


def test_build_qa_prompt_represents_output_schema_fields():
    prompt = build_qa_prompt("irrelevant question", [])

    for field in (
        "intent",
        "metric",
        "column_hint",
        "period_hint",
        "from_period_hint",
        "to_period_hint",
    ):
        assert field in prompt


def test_build_qa_prompt_instructs_json_only_output():
    prompt = build_qa_prompt("irrelevant question", [])

    assert "Return JSON only" in prompt


def test_build_qa_prompt_prohibits_answering_calculating_or_inventing_values():
    prompt = build_qa_prompt("irrelevant question", [])

    assert "answer the user's question" in prompt
    assert "calculate" in prompt
    assert "NEVER invent a value for any hint" in prompt
    assert "NEVER convert a hint into a numeric result" in prompt


def test_build_qa_prompt_explains_python_resolves_ambiguity():
    prompt = build_qa_prompt("irrelevant question", [])

    assert "NEVER resolve ambiguity" in prompt
    assert "resolution is performed later by Python" in prompt


def test_build_qa_prompt_handles_empty_column_names_without_error():
    prompt = build_qa_prompt("What is the total?", [])

    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_build_qa_prompt_is_pure_no_side_effects():
    columns = ["Revenue", "Month"]
    before = list(columns)

    build_qa_prompt("What was the value in March?", columns)

    assert columns == before


def test_build_qa_prompt_period_change_uses_from_and_to_hints():
    prompt = build_qa_prompt("Compare March to April", ["Revenue"])

    assert "from_period_hint" in prompt
    assert "to_period_hint" in prompt


def test_build_qa_prompt_does_not_include_extra_keys_instruction_is_present():
    prompt = build_qa_prompt("irrelevant question", [])

    assert "extra keys" in prompt
