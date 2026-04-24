from src.known_gap.domain.services.llm_provider import LLMProvider
from src.known_gap.shared.exceptions.base import TransientException


class FallbackLLMProvider(LLMProvider):
    """Tries the primary provider first; on transient failure only, falls
    back to the secondary. Permanent errors propagate — they indicate a
    request that would fail the same way on any provider.
    """

    def __init__(self, primary: LLMProvider, fallback: LLMProvider) -> None:
        self._primary = primary
        self._fallback = fallback

    async def complete(self, system: str, user: str, max_tokens: int) -> str:
        try:
            return await self._primary.complete(system, user, max_tokens)
        except TransientException:
            return await self._fallback.complete(system, user, max_tokens)
