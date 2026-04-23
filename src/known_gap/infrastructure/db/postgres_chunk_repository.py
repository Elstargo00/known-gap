from collections.abc import Sequence

import asyncpg

from src.known_gap.domain.models.chunk import Chunk
from src.known_gap.domain.repositories.chunk_repository import ChunkRepository


class PostgresChunkRepository(ChunkRepository):
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def save_many(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return
        rows = [
            (
                chunk.id,
                chunk.document_id,
                chunk.index,
                chunk.content,
                chunk.embedding,
            )
            for chunk in chunks
        ]
        async with self._pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO chunks (id, document_id, chunk_index, content, embedding)
                VALUES ($1, $2, $3, $4, $5)
                """,
                rows,
            )
