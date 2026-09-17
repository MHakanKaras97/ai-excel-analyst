import os

import pytest

from src.ai_interpreter import interpret
from src.comparison_ai_payload import build_comparison_ai_payload
from src.comparison_prompt_builder import build_comparison_prompt
from src.gemini_provider import GeminiProvider

pytestmark = pytest.mark.integration

pytest.importorskip("google.genai", reason="google-genai SDK is not installed")

if not os.environ.get("GEMINI_API_KEY"):
    pytest.skip("GEMINI_API_KEY is not set", allow_module_level=True)


def test_gemini_comparison_provider_live_call_returns_parseable_insight():
    provider = GeminiProvider()
    comparison_result = {
        "found": True, "reason": None, "comparison_type": "file_value_comparison",
        "metric": "sum", "column": "revenue", "period": None,
        "file_a": {"file_id": "internal-id-a", "display_name": "January", "role": "previous"},
        "file_b": {"file_id": "internal-id-b", "display_name": "February", "role": "current"},
        "previous": 100000.0, "current": 130000.0, "absolute_change": 30000.0,
        "percentage_change": 30.0, "is_valid": True, "extra": {},
    }
    ai_payload = build_comparison_ai_payload(comparison_result)

    result = interpret(ai_payload, provider, prompt_builder=build_comparison_prompt)

    assert result["is_valid"] is True
    assert result["insight"] is not None
    assert isinstance(result["insight"]["summary"], str)
    assert len(result["insight"]["summary"]) > 0
