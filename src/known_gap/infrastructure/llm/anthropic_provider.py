import anthropic
from anthropic import AsyncAnthropic
from anthropic.types import TextBlock

from src.known_gap.domain.services.llm_provider import LLMProvider
from src.known_gap.shared.exceptions.base import PermanentException, TransientException


class AnthropicLLMProvider(LLMProvider):
    def __init__(self, api_key: str, model: str) -> None:
        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model

    async def complete(self, system: str, user: str, max_tokens: int) -> str:
        try:
            message = await self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except anthropic.APIConnectionError as e:
            raise self._transient(e) from e
        except anthropic.RateLimitError as e:
            raise self._transient(e) from e
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                raise self._transient(e) from e
            raise self._permanent(e) from e
        except anthropic.APIError as e:
            raise self._permanent(e) from e

        return "".join(block.text for block in message.content if isinstance(block, TextBlock))

    def _transient(self, error: Exception) -> TransientException:
        return TransientException(
            message=f"Anthropic request failed transiently: {error}",
            error_code="LLM_PROVIDER_ERROR",
            details={"provider": "anthropic", "error_type": type(error).__name__},
        )

    def _permanent(self, error: Exception) -> PermanentException:
        return PermanentException(
            message=f"Anthropic request failed: {error}",
            error_code="LLM_PROVIDER_ERROR",
            details={"provider": "anthropic", "error_type": type(error).__name__},
        )
