from abc import ABC, abstractmethod
from collections.abc import Sequence

from src.known_gap.domain.models.chunk import Chunk


class ChunkRepository(ABC):
    @abstractmethod
    async def save_many(self, chunks: Sequence[Chunk]) -> None: ...
