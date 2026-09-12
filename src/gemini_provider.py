import os

from src.ai_provider import AIProviderError

DEFAULT_MODEL = "gemini-flash-lite-latest"


class GeminiProvider:
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL, client=None):
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._model = model
        self._client = client

    def _build_client(self):
        try:
            from google import genai
        except ImportError as exc:
            raise AIProviderError("provider_unavailable", str(exc)) from exc

        return genai.Client(api_key=self._api_key)

    def generate(self, prompt: str) -> str:
        if not self._api_key:
            raise AIProviderError("missing_api_key")

        client = self._client or self._build_client()

        try:
            response = client.models.generate_content(model=self._model, contents=prompt)
        except AIProviderError:
            raise
        except Exception as exc:
            raise AIProviderError("provider_error", str(exc)) from exc

        return getattr(response, "text", "") or ""
