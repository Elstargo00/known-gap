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


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    mode: AskMode
    known_concepts: list[ConceptView] = Field(default_factory=list)
    unknown_concepts: list[ConceptView] = Field(default_factory=list)
