from uuid import UUID

from pydantic import BaseModel


class IngestResponse(BaseModel):
    document_id: UUID
    filename: str
    chunk_count: int
