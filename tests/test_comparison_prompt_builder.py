from src.comparison_prompt_builder import build_comparison_prompt


def _payload(**overrides):
    payload = {
        "found": True, "reason": None, "comparison_type": "file_value_comparison",
        "metric": "sum", "column": "TotalPrice", "period": None,
        "file_a": {"display_name": "January", "role": "previous"},
        "file_b": {"display_name": "February", "role": "current"},
        "previous": 100000.0, "current": 130000.0, "absolute_change": 30000.0,
        "percentage_change": 30.0, "is_valid": True, "extra": {},
    }
    payload.update(overrides)
    return payload


def test_build_comparison_prompt_returns_a_string():
    prompt = build_comparison_prompt(_payload())

    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_build_comparison_prompt_contains_comparison_specific_instructions():
    prompt = build_comparison_prompt(_payload())

    assert "comparison between two files/periods" in prompt
    assert "not a single dataset" in prompt


def test_build_comparison_prompt_embeds_payload_verbatim():
    prompt = build_comparison_prompt(_payload())

    assert "January" in prompt
    assert "February" in prompt
    assert "100000.0" in prompt
    assert "30000.0" in prompt


def test_build_comparison_prompt_explains_previous_current_semantics():
    prompt = build_comparison_prompt(_payload())

    assert "previous" in prompt and "file_a" in prompt
    assert "current" in prompt and "file_b" in prompt
    assert "never infer or reassign which file is which" in prompt


def test_build_comparison_prompt_explains_file_identity_fields():
    prompt = build_comparison_prompt(_payload())

    assert "display_name" in prompt
    assert "role" in prompt
    assert "refer to them by their supplied display name or role" in prompt


def test_build_comparison_prompt_explains_absolute_and_percentage_change_grounding():
    prompt = build_comparison_prompt(_payload())

    assert "already-computed deterministic facts" in prompt
    assert "must NEVER recalculate them" in prompt
    assert "recompute them from previous/current" in prompt
    assert "alter them" in prompt
    assert "round them differently" in prompt
    assert "derive any new numeric value" in prompt


def test_build_comparison_prompt_explains_trend_label_guidance():
    prompt = build_comparison_prompt(_payload())

    assert "trend labels such as" in prompt
    assert "Never invent a numeric value to accompany a trend-label comparison" in prompt
    assert "never treat a trend label as if it were a number" in prompt


def test_build_comparison_prompt_prohibits_inferring_unsupported_relationships():
    prompt = build_comparison_prompt(_payload())

    assert "Do not infer a comparison relationship, file relationship, period, or metric" in prompt


def test_build_comparison_prompt_prohibits_calculation_and_invention():
    prompt = build_comparison_prompt(_payload())

    assert "Do not calculate, estimate, round, or invent any numeric value not present in the data" in prompt


def test_build_comparison_prompt_instructs_json_only_output():
    prompt = build_comparison_prompt(_payload())

    assert "Respond with ONLY valid JSON" in prompt


def test_build_comparison_prompt_output_schema_matches_existing_insight_shape():
    prompt = build_comparison_prompt(_payload())

    for field in ("summary", "key_insights", "trend_interpretation", "warnings", "recommendations"):
        assert field in prompt


def test_build_comparison_prompt_does_not_introduce_new_output_fields():
    from src.comparison_prompt_builder import OUTPUT_SCHEMA_DESCRIPTION

    assert "comparison_interpretation" not in OUTPUT_SCHEMA_DESCRIPTION
    assert "absolute_change" not in OUTPUT_SCHEMA_DESCRIPTION
    assert "percentage_change" not in OUTPUT_SCHEMA_DESCRIPTION


def test_build_comparison_prompt_does_not_mutate_input():
    payload = _payload()
    before = dict(payload)

    build_comparison_prompt(payload)

    assert payload == before


def test_build_comparison_prompt_handles_trend_comparison_string_values():
    payload = _payload(
        comparison_type="file_trend_comparison",
        previous="increasing", current="decreasing",
        absolute_change=None, percentage_change=None, metric=None, column="TotalPrice",
    )

    prompt = build_comparison_prompt(payload)

    assert "increasing" in prompt
    assert "decreasing" in prompt


def test_build_comparison_prompt_handles_failure_shaped_payload_without_crashing():
    payload = {
        "found": False, "reason": "file_not_found", "comparison_type": "file_value_comparison",
        "metric": None, "column": None, "period": None,
        "file_a": None, "file_b": None,
        "previous": None, "current": None, "absolute_change": None,
        "percentage_change": None, "is_valid": None, "extra": {},
    }

    prompt = build_comparison_prompt(payload)

    assert isinstance(prompt, str)
    assert "file_not_found" in prompt
