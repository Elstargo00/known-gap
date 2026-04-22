from dataclasses import dataclass


@dataclass(frozen=True)
class IngestDocumentCommand:
    filename: str
    content_type: str
    data: bytes
