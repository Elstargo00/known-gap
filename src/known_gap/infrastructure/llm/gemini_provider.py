from google import genai
from google.genai import errors as genai_errors
from google.genai.types import GenerateContentConfig

from src.known_gap.domain.services.llm_provider import LLMProvider
from src.known_gap.shared.exceptions.base import PermanentException, TransientException


class GeminiLLMProvider(LLMProvider):
    def __init__(self, api_key: str, model: str) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model

    async def complete(self, system: str, user: str, max_tokens: int) -> str:
        try:
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=user,
                config=GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=max_tokens,
                ),
            )
        except genai_errors.ServerError as e:
            raise self._transient(e) from e
        except genai_errors.ClientError as e:
            raise self._permanent(e) from e
        except genai_errors.APIError as e:
            raise self._permanent(e) from e

        return response.text or ""

    def _transient(self, error: Exception) -> TransientException:
        return TransientException(
            message=f"Gemini request failed transiently: {error}",
            error_code="LLM_PROVIDER_ERROR",
            details={"provider": "gemini", "error_type": type(error).__name__},
        )

    def _permanent(self, error: Exception) -> PermanentException:
        return PermanentException(
            message=f"Gemini request failed: {error}",
            error_code="LLM_PROVIDER_ERROR",
            details={"provider": "gemini", "error_type": type(error).__name__},
        )
