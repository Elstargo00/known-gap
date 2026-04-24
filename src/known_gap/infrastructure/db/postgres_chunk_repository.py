from collections.abc import Sequence

import asyncpg

from src.known_gap.domain.models.chunk import Chunk
from src.known_gap.domain.models.retrieved_chunk import RetrievedChunk
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

    async def search_similar(
        self,
        query_embedding: list[float],
        k: int,
    ) -> list[RetrievedChunk]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    c.id           AS chunk_id,
                    c.document_id  AS document_id,
                    c.content      AS content,
                    d.filename     AS filename,
                    1 - (c.embedding <=> $1) AS similarity_score
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.embedding IS NOT NULL
                ORDER BY c.embedding <=> $1
                LIMIT $2
                """,
                query_embedding,
                k,
            )
        return [
            RetrievedChunk(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                content=row["content"],
                document_filename=row["filename"],
                similarity_score=float(row["similarity_score"]),
            )
            for row in rows
        ]
