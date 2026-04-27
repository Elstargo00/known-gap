from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class IngestDocumentCommand:
    """Per-user ingest. The `user_id` only scopes graph operations —
    document chunks themselves remain global."""

    user_id: UUID
    filename: str
    content_type: str
    data: bytes
