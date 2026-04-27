import asyncio
import logging
from datetime import UTC, datetime
from uuid import UUID, uuid4

from src.known_gap.application.services.chunker import Chunker
from src.known_gap.application.services.concept_extractor import ConceptExtractor
from src.known_gap.application.services.graph_expander import GraphExpander
from src.known_gap.application.use_cases.ingest_document.command import IngestDocumentCommand
from src.known_gap.application.use_cases.ingest_document.result import IngestDocumentResult
from src.known_gap.domain.models.chunk import Chunk
from src.known_gap.domain.models.document import Document
from src.known_gap.domain.repositories.chunk_repository import ChunkRepository
from src.known_gap.domain.repositories.document_repository import DocumentRepository
from src.known_gap.domain.repositories.user_graph_repository import UserGraphRepository
from src.known_gap.domain.services.document_parser import DocumentParser
from src.known_gap.domain.services.embedding_provider import EmbeddingProvider
from src.known_gap.shared.exceptions.base import PermanentException

logger = logging.getLogger(__name__)

# Cap the text we send to the concept extractor on ingestion. Beyond a
# few thousand chars the marginal value of more text drops fast and the
# call gets expensive — Haiku still handles long inputs but we don't
# need them for graph seeding.
INGEST_CONCEPT_TEXT_CAP = 16_000


class IngestDocumentHandler:
    """Ingests a document end-to-end: parse → chunk → embed → persist
    document & chunks → seed the user's knowledge graph from the same
    text → kick off background graph expansion for densification.

    Graph seeding and expansion are guarded so a failure here never
    fails the ingestion API call — the corpus side of the system is the
    promise; the graph side is best-effort."""

    def __init__(
        self,
        parser: DocumentParser,
        chunker: Chunker,
        embedder: EmbeddingProvider,
        documents: DocumentRepository,
        chunks: ChunkRepository,
        concept_extractor: ConceptExtractor,
        graph: UserGraphRepository,
        graph_expander: GraphExpander,
        initial_score: int = 50,
    ) -> None:
        self._parser = parser
        self._chunker = chunker
        self._embedder = embedder
        self._documents = documents
        self._chunks = chunks
        self._concept_extractor = concept_extractor
        self._graph = graph
        self._graph_expander = graph_expander
        self._initial_score = initial_score

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

        # Best-effort graph seeding: extract concepts from the (capped)
        # full text, upsert them with the configured initial score in the
        # user's graph, and fire off the graph expander in the background.
        try:
            await self._seed_graph(command.user_id, text)
        except Exception:  # noqa: BLE001 — ingest must succeed even if the graph side fails
            logger.exception(
                "graph seeding failed during ingest",
                extra={
                    "user_id": str(command.user_id),
                    "filename": command.filename,
                },
            )

        return IngestDocumentResult(
            document_id=document.id,
            filename=document.filename,
            chunk_count=len(persisted_chunks),
        )

    async def _seed_graph(self, user_id: UUID, text: str) -> None:
        snippet = text[:INGEST_CONCEPT_TEXT_CAP]
        if not snippet.strip():
            return
        mentions = await self._concept_extractor.extract(snippet)
        if not mentions:
            return
        await self._graph.upsert_concepts(
            user_id, mentions, initial_score=self._initial_score
        )
        seed_canonicals = [m.canonical_name for m in mentions]
        if seed_canonicals:
            asyncio.create_task(self._graph_expander.expand(user_id, seed_canonicals))
