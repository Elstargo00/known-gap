from abc import ABC, abstractmethod
from collections.abc import Sequence


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]: ...
