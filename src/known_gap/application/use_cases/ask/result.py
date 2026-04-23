from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class AskSource:
    chunk_id: UUID
    document_id: UUID
    filename: str
    content_preview: str
    similarity_score: float


@dataclass(frozen=True)
class AskResult:
    answer: str
    sources: list[AskSource]
    mode: str
