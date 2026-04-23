from src.known_gap.config.settings import Settings
from src.known_gap.domain.services.embedding_provider import EmbeddingProvider
from src.known_gap.infrastructure.embeddings.voyage_provider import VoyageEmbeddingProvider


class EmbeddingProviderFactory:
    @staticmethod
    def from_settings(settings: Settings) -> EmbeddingProvider:
        provider = settings.embedding_provider.lower()
        if provider == "voyage":
            if not settings.voyage_api_key:
                raise ValueError("VOYAGE_API_KEY is required for the voyage provider")
            return VoyageEmbeddingProvider(
                api_key=settings.voyage_api_key,
                model=settings.embedding_model,
                dimension=settings.embedding_dimension,
                batch_size=settings.embedding_batch_size,
            )
        raise ValueError(f"Unknown embedding provider: {provider!r}")
