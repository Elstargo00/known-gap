from typing import Annotated

from fastapi import Depends, Request

from src.known_gap.application.services.chunker import Chunker
from src.known_gap.application.use_cases.ingest_document.handler import IngestDocumentHandler
from src.known_gap.config.settings import Settings, get_settings
from src.known_gap.infrastructure.db.postgres_chunk_repository import PostgresChunkRepository
from src.known_gap.infrastructure.db.postgres_document_repository import (
    PostgresDocumentRepository,
)
from src.known_gap.infrastructure.embeddings.factory import EmbeddingProviderFactory
from src.known_gap.infrastructure.parsing.document_parser import DispatchingDocumentParser

SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_ingest_handler(
    request: Request,
    settings: SettingsDep,
) -> IngestDocumentHandler:
    pool = request.app.state.pool
    return IngestDocumentHandler(
        parser=DispatchingDocumentParser(),
        chunker=Chunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        ),
        embedder=EmbeddingProviderFactory.from_settings(settings),
        documents=PostgresDocumentRepository(pool),
        chunks=PostgresChunkRepository(pool),
    )


IngestHandlerDep = Annotated[IngestDocumentHandler, Depends(get_ingest_handler)]
