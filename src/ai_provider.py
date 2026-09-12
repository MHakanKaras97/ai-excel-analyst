from typing import Protocol


class AIProviderError(Exception):
    def __init__(self, reason: str, message: str | None = None):
        super().__init__(message or reason)
        self.reason = reason


class AIProvider(Protocol):
    def generate(self, prompt: str) -> str:
        ...
