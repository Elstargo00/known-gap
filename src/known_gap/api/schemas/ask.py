from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    mode: Literal["normal"] = "normal"


class Source(BaseModel):
    chunk_id: UUID
    document_id: UUID
    filename: str
    content_preview: str
    similarity_score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    mode: str
