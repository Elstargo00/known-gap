from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


@dataclass(frozen=True)
class Document:
    id: UUID
    filename: str
    content_type: str
    byte_size: int
    ingested_at: datetime
