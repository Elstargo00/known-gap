from datetime import UTC, datetime
from uuid import uuid4

from src.known_gap.application.services.chunker import Chunker
from src.known_gap.application.use_cases.ingest_document.command import IngestDocumentCommand
from src.known_gap.application.use_cases.ingest_document.result import IngestDocumentResult
from src.known_gap.domain.models.chunk import Chunk
from src.known_gap.domain.models.document import Document
from src.known_gap.domain.repositories.chunk_repository import ChunkRepository
from src.known_gap.domain.repositories.document_repository import DocumentRepository
from src.known_gap.domain.services.document_parser import DocumentParser
from src.known_gap.domain.services.embedding_provider import EmbeddingProvider
from src.known_gap.shared.exceptions.base import PermanentException


class IngestDocumentHandler:
    def __init__(
        self,
        parser: DocumentParser,
        chunker: Chunker,
        embedder: EmbeddingProvider,
        documents: DocumentRepository,
        chunks: ChunkRepository,
    ) -> None:
        self._parser = parser
        self._chunker = chunker
        self._embedder = embedder
        self._documents = documents
        self._chunks = chunks

    async def execute(self, command: IngestDocumentCommand) -> IngestDocumentResult:
        text = self._parser.parse(command.filename, command.data)
        text_chunks = self._chunker.chunk(text)
        if not text_chunks:
            raise PermanentException(
                message="Document produced no extractable text",
                error_code="EMPTY_DOCUMENT",
                details={"filename": command.filename},
            )

        embeddings = await self._embedder.embed_documents([chunk.content for chunk in text_chunks])

        document = Document(
            id=uuid4(),
            filename=command.filename,
            content_type=command.content_type,
            byte_size=len(command.data),
            ingested_at=datetime.now(UTC),
        )
        await self._documents.save(document)

        persisted_chunks = [
            Chunk(
                id=uuid4(),
                document_id=document.id,
                index=text_chunk.index,
                content=text_chunk.content,
                embedding=embedding,
            )
            for text_chunk, embedding in zip(text_chunks, embeddings, strict=True)
        ]
        await self._chunks.save_many(persisted_chunks)

        return IngestDocumentResult(
            document_id=document.id,
            filename=document.filename,
            chunk_count=len(persisted_chunks),
        )
