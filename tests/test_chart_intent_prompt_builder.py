from src.chart_intent_prompt_builder import build_chart_intent_prompt

SUPPORTED_INTENTS = (
    "trend_chart",
    "numeric_summary_chart",
    "period_change_chart",
    "missing_values_chart",
    "unsupported",
)

SUPPORTED_METRICS = ("sum", "mean", "median", "min", "max", "std", "count")


def test_build_chart_intent_prompt_returns_a_string():
    prompt = build_chart_intent_prompt("Show the TotalPrice trend.", ["TotalPrice", "Month"])

    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_build_chart_intent_prompt_includes_the_user_question():
    prompt = build_chart_intent_prompt("Plot TotalPrice over time.", ["TotalPrice", "Month"])

    assert "Plot TotalPrice over time." in prompt


def test_build_chart_intent_prompt_includes_supplied_column_names():
    prompt = build_chart_intent_prompt("Show missing values.", ["Revenue", "Region", "Date"])

    assert "Revenue" in prompt
    assert "Region" in prompt
    assert "Date" in prompt


def test_build_chart_intent_prompt_represents_all_five_intents():
    prompt = build_chart_intent_prompt("irrelevant question", [])

    for intent in SUPPORTED_INTENTS:
        assert intent in prompt


def test_build_chart_intent_prompt_represents_metric_enum():
    prompt = build_chart_intent_prompt("irrelevant question", [])

    for metric in SUPPORTED_METRICS:
        assert metric in prompt


def test_build_chart_intent_prompt_represents_exact_output_schema_fields():
    prompt = build_chart_intent_prompt("irrelevant question", [])

    for field in (
        "intent",
        "column_hint",
        "metric",
        "period_hint",
        "from_period_hint",
        "to_period_hint",
    ):
        assert field in prompt


def test_build_chart_intent_prompt_instructs_json_only_output():
    prompt = build_chart_intent_prompt("irrelevant question", [])

    assert "Return JSON only" in prompt


def test_build_chart_intent_prompt_prohibits_calculation_and_resolution():
    prompt = build_chart_intent_prompt("irrelevant question", [])

    assert "calculate anything" in prompt
    assert "resolve column names" in prompt
    assert "resolve periods" in prompt
    assert "select an actual dataframe column" in prompt
    assert "generate a chart" in prompt
    assert "NEVER invent a value for any hint" in prompt
    assert "NEVER resolve a hint to an actual dataset column or period" in prompt


def test_build_chart_intent_prompt_does_not_instruct_choosing_a_column():
    prompt = build_chart_intent_prompt("irrelevant question", ["TotalPrice", "Region"])

    assert "context only" in prompt
    assert "do not choose one" in prompt


def test_build_chart_intent_prompt_explains_hints_are_raw_user_wording():
    prompt = build_chart_intent_prompt("irrelevant question", [])

    assert "exactly/as closely as practical" in prompt
    assert "do not convert it into an actual dataset column name" in prompt


def test_build_chart_intent_prompt_handles_empty_column_names_without_error():
    prompt = build_chart_intent_prompt("Show missing values.", [])

    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_build_chart_intent_prompt_is_pure_no_side_effects():
    columns = ["TotalPrice", "Month"]
    before = list(columns)

    build_chart_intent_prompt("Show the TotalPrice trend.", columns)

    assert columns == before


def test_build_chart_intent_prompt_period_change_uses_from_and_to_hints():
    prompt = build_chart_intent_prompt("Plot the change from January to March.", ["TotalPrice"])

    assert "from_period_hint" in prompt
    assert "to_period_hint" in prompt


def test_build_chart_intent_prompt_does_not_call_a_provider():
    # Purity check: the function accepts no provider argument and performs
    # no I/O — calling it repeatedly with the same input is deterministic.
    first = build_chart_intent_prompt("Show the TotalPrice trend.", ["TotalPrice"])
    second = build_chart_intent_prompt("Show the TotalPrice trend.", ["TotalPrice"])

    assert first == second
