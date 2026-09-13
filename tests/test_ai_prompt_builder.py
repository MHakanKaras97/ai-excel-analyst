import json

from src.ai_prompt_builder import build_prompt


def test_build_prompt_includes_system_instructions():
    prompt = build_prompt({})

    assert "Only reference numbers that appear verbatim" in prompt
    assert "Do not calculate, estimate, round, or invent" in prompt


def test_build_prompt_includes_output_schema():
    prompt = build_prompt({})

    assert "summary" in prompt
    assert "key_insights" in prompt
    assert "trend_interpretation" in prompt
    assert "warnings" in prompt
    assert "recommendations" in prompt


def test_build_prompt_embeds_analytics_data_verbatim():
    payload = {"numeric_summary": {"columns": [{"name": "amount", "sum": 12345.0}]}}

    prompt = build_prompt(payload)

    assert "amount" in prompt
    assert "12345.0" in prompt
    assert json.dumps(payload, indent=2) in prompt


def test_build_prompt_handles_empty_payload():
    prompt = build_prompt({})

    assert "{}" in prompt
    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_build_prompt_is_pure_no_side_effects():
    payload = {"a": 1}
    before = dict(payload)

    build_prompt(payload)

    assert payload == before


def test_build_prompt_instructs_prioritization_over_restating_every_metric():
    prompt = build_prompt({})

    assert "Do not restate every metric in the data" in prompt
    assert "2-3 most decision-relevant findings" in prompt


def test_build_prompt_instructs_summary_as_concise_executive_takeaway():
    prompt = build_prompt({})

    assert "1-2 sentences" in prompt
    assert "single most important takeaway" in prompt


def test_build_prompt_instructs_key_insights_to_explain_why_not_just_the_number():
    prompt = build_prompt({})

    assert "explain why each finding matters, not just the number" in prompt


def test_build_prompt_instructs_trend_interpretation_on_direction_and_consistency():
    prompt = build_prompt({})

    assert "describe direction and consistency using the trend data" in prompt


def test_build_prompt_instructs_warnings_limited_to_data_quality_concerns():
    prompt = build_prompt({})

    assert 'Use "warnings" only for visible data-quality or reliability concerns' in prompt


def test_build_prompt_instructs_recommendations_tied_to_a_stated_insight_or_warning():
    prompt = build_prompt({})

    assert "Every recommendation must be an action that follows directly from a specific insight or warning" in prompt


def test_build_prompt_instructs_observations_kept_separate_from_recommendations():
    prompt = build_prompt({})

    assert "Keep observations" in prompt
    assert "separate from actions" in prompt


def test_build_prompt_preserves_existing_grounding_rules_alongside_new_instructions():
    prompt = build_prompt({})

    assert "Only reference numbers that appear verbatim" in prompt
    assert "Do not calculate, estimate, round, or invent" in prompt
    assert "Respond with ONLY valid JSON matching the schema below" in prompt


def test_build_prompt_instructs_no_unsupported_value_judgments():
    prompt = build_prompt({})

    assert (
        "Do not characterize a value as high, low, good, bad, concerning, or risky "
        "unless the analytics data provides an explicit basis for that judgment, "
        "such as a threshold, benchmark, comparison, or trend."
    ) in prompt
    assert "describe the value neutrally without assigning business meaning" in prompt


def test_build_prompt_clarifies_evaluative_label_restriction_does_not_suppress_other_rules():
    prompt = build_prompt({})

    assert (
        'This restriction applies only to evaluative labels (e.g. "low", "bad", "concerning") '
        "— it does not exempt you from prioritizing the most decision-relevant findings, "
        "explaining trend direction and consistency, or providing grounded, actionable "
        "recommendations tied to a stated insight or warning."
    ) in prompt


def test_build_prompt_instructs_interpreting_patterns_without_a_benchmark():
    prompt = build_prompt({})

    assert (
        "Interpret the observed patterns and explain why they matter even when no "
        "benchmark or threshold exists. You may describe trends, volatility, changes, "
        "consistency, and other patterns neutrally and recommend actions that address "
        'those patterns. Avoid unsupported evaluative labels such as "low", "bad", '
        '"concerning", or "risky" unless the analytics data provides an explicit basis '
        "for them."
    ) in prompt


def test_build_prompt_instructs_trend_interpretation_as_words_not_counts():
    prompt = build_prompt({})

    assert (
        'trend_interpretation must describe the pattern in words (e.g. "alternates '
        'with no sustained direction", "consistently trending upward", or '
        '"consistently trending downward"). Do not restate the increase/decrease/'
        "comparison counts themselves inside trend_interpretation; those counts are "
        "already available in the analytics data."
    ) in prompt


def test_build_prompt_still_embeds_analytics_payload_verbatim_with_new_instructions():
    payload = {
        "trend": {"trend": "volatile", "increase_count": 13, "decrease_count": 18},
        "numeric_summary": {"columns": [{"name": "TotalPrice", "sum": 453850.31}]},
    }

    prompt = build_prompt(payload)

    assert json.dumps(payload, indent=2) in prompt
    assert "453850.31" in prompt
    assert "Do not restate every metric in the data" in prompt
