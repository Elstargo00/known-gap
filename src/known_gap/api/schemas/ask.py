from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

AskMode = Literal["normal", "learning", "concise"]


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    mode: AskMode = "normal"


class Source(BaseModel):
    chunk_id: UUID
    document_id: UUID
    filename: str
    content_preview: str
    similarity_score: float


class ConceptView(BaseModel):
    canonical_name: str
    display_name: str
    known_score: int = 0


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    mode: AskMode
    known_concepts: list[ConceptView] = Field(default_factory=list)
    unknown_concepts: list[ConceptView] = Field(default_factory=list)
    # Canonical names that the answer was masked on for learning-mode
    # cloze fill-in-the-blanks. Frontend renders <cloze ...> tags
    # embedded inline in `answer`. Empty when mode != "learning".
    cloze_concepts: list[str] = Field(default_factory=list)
