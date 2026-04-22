from abc import ABC, abstractmethod

from src.known_gap.domain.models.document import Document


class DocumentRepository(ABC):
    @abstractmethod
    async def save(self, document: Document) -> None: ...
