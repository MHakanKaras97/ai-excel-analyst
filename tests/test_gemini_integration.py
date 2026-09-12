import os

import pytest

from src.ai_interpreter import interpret
from src.gemini_provider import GeminiProvider

pytestmark = pytest.mark.integration

pytest.importorskip("google.genai", reason="google-genai SDK is not installed")

if not os.environ.get("GEMINI_API_KEY"):
    pytest.skip("GEMINI_API_KEY is not set", allow_module_level=True)


def test_gemini_provider_live_call_returns_parseable_insight():
    provider = GeminiProvider()
    payload = {
        "numeric_summary": {
            "columns": [{"name": "revenue", "count": 4, "sum": 400000.0, "mean": 100000.0}]
        }
    }

    result = interpret(payload, provider)

    assert result["is_valid"] is True
    assert result["insight"] is not None
    assert isinstance(result["insight"]["summary"], str)
