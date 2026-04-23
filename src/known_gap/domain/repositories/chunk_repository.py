from abc import ABC, abstractmethod
from collections.abc import Sequence

from src.known_gap.domain.models.chunk import Chunk
from src.known_gap.domain.models.retrieved_chunk import RetrievedChunk


class ChunkRepository(ABC):
    @abstractmethod
    async def save_many(self, chunks: Sequence[Chunk]) -> None: ...

    @abstractmethod
    async def search_similar(
        self,
        query_embedding: list[float],
        k: int,
    ) -> list[RetrievedChunk]: ...
