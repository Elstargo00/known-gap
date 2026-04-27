from dataclasses import dataclass, field
from uuid import UUID


@dataclass(frozen=True)
class AskSource:
    chunk_id: UUID
    document_id: UUID
    filename: str
    content_preview: str
    similarity_score: float


@dataclass(frozen=True)
class AskConcept:
    canonical_name: str
    display_name: str
    known_score: int = 0


@dataclass(frozen=True)
class AskResult:
    answer: str
    sources: list[AskSource]
    mode: str
    known_concepts: list[AskConcept] = field(default_factory=list)
    unknown_concepts: list[AskConcept] = field(default_factory=list)
    cloze_concepts: list[str] = field(default_factory=list)
