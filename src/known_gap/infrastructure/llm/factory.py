from src.known_gap.config.settings import Settings
from src.known_gap.domain.services.llm_provider import LLMProvider
from src.known_gap.infrastructure.llm.anthropic_provider import AnthropicLLMProvider
from src.known_gap.infrastructure.llm.fallback_provider import FallbackLLMProvider
from src.known_gap.infrastructure.llm.gemini_provider import GeminiLLMProvider


class LLMProviderFactory:
    @staticmethod
    def from_settings(settings: Settings) -> LLMProvider:
        return LLMProviderFactory._build(settings, settings.llm_primary_model)

    @staticmethod
    def for_concept_extraction(settings: Settings) -> LLMProvider:
        return LLMProviderFactory._build(settings, settings.concept_extraction_model)

    @staticmethod
    def for_graph_expert(settings: Settings) -> LLMProvider:
        return LLMProviderFactory._build(settings, settings.graph_expert_model)

    @staticmethod
    def _build(settings: Settings, primary_model: str) -> LLMProvider:
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for the primary LLM provider")

        primary = AnthropicLLMProvider(
            api_key=settings.anthropic_api_key,
            model=primary_model,
        )
        if not settings.gemini_api_key:
            return primary

        fallback = GeminiLLMProvider(
            api_key=settings.gemini_api_key,
            model=settings.llm_fallback_model,
        )
        return FallbackLLMProvider(primary=primary, fallback=fallback)
