from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class Chunk:
    id: UUID
    document_id: UUID
    index: int
    content: str
    embedding: list[float]
