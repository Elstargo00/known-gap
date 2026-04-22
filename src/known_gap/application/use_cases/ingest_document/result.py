from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class IngestDocumentResult:
    document_id: UUID
    filename: str
    chunk_count: int
