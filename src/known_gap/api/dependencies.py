from typing import Annotated

from fastapi import Depends, Request

from src.known_gap.application.services.chunker import Chunker
from src.known_gap.application.services.prompt_builder import PromptBuilder
from src.known_gap.application.use_cases.ask.handler import AskHandler
from src.known_gap.application.use_cases.ingest_document.handler import IngestDocumentHandler
from src.known_gap.config.settings import Settings, get_settings
from src.known_gap.infrastructure.db.postgres_chunk_repository import PostgresChunkRepository
from src.known_gap.infrastructure.db.postgres_document_repository import (
    PostgresDocumentRepository,
)
from src.known_gap.infrastructure.embeddings.factory import EmbeddingProviderFactory
from src.known_gap.infrastructure.llm.factory import LLMProviderFactory
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


def get_ask_handler(
    request: Request,
    settings: SettingsDep,
) -> AskHandler:
    pool = request.app.state.pool
    return AskHandler(
        embedder=EmbeddingProviderFactory.from_settings(settings),
        chunks=PostgresChunkRepository(pool),
        llm=LLMProviderFactory.from_settings(settings),
        prompt_builder=PromptBuilder(),
        top_k=settings.top_k,
        answer_max_tokens=settings.answer_max_tokens,
    )


IngestHandlerDep = Annotated[IngestDocumentHandler, Depends(get_ingest_handler)]
AskHandlerDep = Annotated[AskHandler, Depends(get_ask_handler)]
