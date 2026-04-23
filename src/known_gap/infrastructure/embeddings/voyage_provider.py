from collections.abc import Sequence
from typing import cast

import voyageai

from src.known_gap.domain.services.embedding_provider import EmbeddingProvider
from src.known_gap.shared.exceptions.base import TransientException


class VoyageEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        api_key: str,
        model: str,
        dimension: int,
        batch_size: int = 64,
    ) -> None:
        self._client = voyageai.AsyncClient(api_key=api_key)
        self._model = model
        self._dimension = dimension
        self._batch_size = batch_size

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        all_vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            all_vectors.extend(await self._embed(batch, input_type="document"))
        return all_vectors

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self._embed([text], input_type="query")
        return vectors[0]

    async def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        try:
            result = await self._client.embed(
                texts,
                model=self._model,
                input_type=input_type,
            )
            return cast(list[list[float]], result.embeddings)
        except Exception as e:
            raise TransientException(
                message=f"Voyage embedding request failed: {e}",
                error_code="EMBEDDING_PROVIDER_ERROR",
                details={"error_type": type(e).__name__, "model": self._model},
            ) from e
