import sys

import pytest

from src.ai_provider import AIProviderError
from src.gemini_provider import GeminiProvider


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeModels:
    def __init__(self, text=None, error: Exception | None = None):
        self._text = text
        self._error = error
        self.last_call = None

    def generate_content(self, model, contents):
        self.last_call = {"model": model, "contents": contents}
        if self._error is not None:
            raise self._error
        return FakeResponse(self._text)


class FakeClient:
    def __init__(self, text=None, error: Exception | None = None):
        self.models = FakeModels(text=text, error=error)


def test_generate_missing_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    provider = GeminiProvider(api_key=None)

    with pytest.raises(AIProviderError) as exc_info:
        provider.generate("prompt")

    assert exc_info.value.reason == "missing_api_key"


def test_generate_uses_explicit_api_key_over_missing_env(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    client = FakeClient(text='{"summary": "ok"}')
    provider = GeminiProvider(api_key="explicit-key", client=client)

    result = provider.generate("prompt")

    assert result == '{"summary": "ok"}'
    assert client.models.last_call["contents"] == "prompt"


def test_generate_falls_back_to_environment_variable(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    client = FakeClient(text="hello")
    provider = GeminiProvider(client=client)

    result = provider.generate("prompt")

    assert result == "hello"


def test_generate_wraps_sdk_exception_as_provider_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    client = FakeClient(error=RuntimeError("network down"))
    provider = GeminiProvider(client=client)

    with pytest.raises(AIProviderError) as exc_info:
        provider.generate("prompt")

    assert exc_info.value.reason == "provider_error"


def test_generate_handles_none_text_response(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    client = FakeClient(text=None)
    provider = GeminiProvider(client=client)

    result = provider.generate("prompt")

    assert result == ""


def test_generate_raises_provider_unavailable_when_sdk_not_installed(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")

    # Force `from google import genai` to fail inside _build_client, regardless
    # of whether google-genai is actually installed in this environment.
    monkeypatch.setitem(sys.modules, "google.genai", None)
    google_module = sys.modules.get("google")
    if google_module is not None:
        monkeypatch.delattr(google_module, "genai", raising=False)

    provider = GeminiProvider()

    with pytest.raises(AIProviderError) as exc_info:
        provider.generate("prompt")

    assert exc_info.value.reason == "provider_unavailable"
