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
