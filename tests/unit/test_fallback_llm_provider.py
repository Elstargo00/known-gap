import pytest

from src.known_gap.domain.services.llm_provider import LLMProvider
from src.known_gap.infrastructure.llm.fallback_provider import FallbackLLMProvider
from src.known_gap.shared.exceptions.base import PermanentException, TransientException


class _StubLLM(LLMProvider):
    def __init__(self, *, response: str | None = None, error: Exception | None = None) -> None:
        self._response = response
        self._error = error
        self.calls = 0

    async def complete(self, system: str, user: str, max_tokens: int) -> str:
        self.calls += 1
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


class TestFallbackLLMProvider:
    async def test_uses_primary_when_primary_succeeds(self) -> None:
        primary = _StubLLM(response="primary answer")
        fallback = _StubLLM(response="fallback answer")

        provider = FallbackLLMProvider(primary=primary, fallback=fallback)
        answer = await provider.complete("sys", "user", 128)

        assert answer == "primary answer"
        assert primary.calls == 1
        assert fallback.calls == 0

    async def test_falls_back_on_transient_error(self) -> None:
        primary = _StubLLM(error=TransientException(message="rate limit"))
        fallback = _StubLLM(response="fallback answer")

        provider = FallbackLLMProvider(primary=primary, fallback=fallback)
        answer = await provider.complete("sys", "user", 128)

        assert answer == "fallback answer"
        assert primary.calls == 1
        assert fallback.calls == 1

    async def test_propagates_permanent_error_from_primary(self) -> None:
        primary = _StubLLM(error=PermanentException(message="bad request"))
        fallback = _StubLLM(response="fallback answer")

        provider = FallbackLLMProvider(primary=primary, fallback=fallback)
        with pytest.raises(PermanentException):
            await provider.complete("sys", "user", 128)

        assert primary.calls == 1
        assert fallback.calls == 0
