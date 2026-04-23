import asyncpg

from src.known_gap.domain.models.document import Document
from src.known_gap.domain.repositories.document_repository import DocumentRepository


class PostgresDocumentRepository(DocumentRepository):
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def save(self, document: Document) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO documents (id, filename, content_type, byte_size, ingested_at)
                VALUES ($1, $2, $3, $4, $5)
                """,
                document.id,
                document.filename,
                document.content_type,
                document.byte_size,
                document.ingested_at,
            )
