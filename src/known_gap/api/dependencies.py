from typing import Annotated
from uuid import UUID

from fastapi import Depends, Header, HTTPException, Request

from src.known_gap.application.services.answer_post_processor import AnswerPostProcessor
from src.known_gap.application.services.chunker import Chunker
from src.known_gap.application.services.concept_extractor import ConceptExtractor
from src.known_gap.application.services.knowledge_classifier import KnowledgeClassifier
from src.known_gap.application.strategies.base import AskStrategy
from src.known_gap.application.strategies.concise import ConciseStrategy
from src.known_gap.application.strategies.learning import LearningStrategy
from src.known_gap.application.strategies.normal import NormalStrategy
from src.known_gap.application.use_cases.ask.handler import AskHandler
from src.known_gap.application.use_cases.ingest_document.handler import IngestDocumentHandler
from src.known_gap.config.settings import Settings, get_settings
from src.known_gap.infrastructure.auth.jwt_verifier import JWTVerifier
from src.known_gap.infrastructure.db.postgres_chunk_repository import PostgresChunkRepository
from src.known_gap.infrastructure.db.postgres_document_repository import (
    PostgresDocumentRepository,
)
from src.known_gap.infrastructure.embeddings.factory import EmbeddingProviderFactory
from src.known_gap.infrastructure.graph.falkordb_user_graph_repository import (
    FalkorDBUserGraphRepository,
)
from src.known_gap.infrastructure.llm.factory import LLMProviderFactory
from src.known_gap.infrastructure.parsing.document_parser import DispatchingDocumentParser
from src.known_gap.shared.exceptions.base import AppException

SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_jwt_verifier(settings: SettingsDep) -> JWTVerifier:
    if not settings.jwt_secret:
        raise HTTPException(
            status_code=500,
            detail="JWT_SECRET is not configured",
        )
    return JWTVerifier(secret=settings.jwt_secret, algorithm=settings.jwt_algorithm)


JWTVerifierDep = Annotated[JWTVerifier, Depends(get_jwt_verifier)]


def get_current_user_id(
    verifier: JWTVerifierDep,
    authorization: Annotated[str | None, Header()] = None,
) -> UUID:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Authorization header must be 'Bearer <token>'",
        )
    token = authorization.removeprefix("Bearer ").strip()
    try:
        return verifier.verify(token)
    except AppException as e:
        raise HTTPException(status_code=e.http_status_code, detail=e.message) from e


CurrentUserId = Annotated[UUID, Depends(get_current_user_id)]


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
    graph_client = request.app.state.graph_client

    generation_llm = LLMProviderFactory.from_settings(settings)
    concept_llm = LLMProviderFactory.for_concept_extraction(settings)

    extractor = ConceptExtractor(
        llm=concept_llm,
        max_tokens=settings.concept_extraction_max_tokens,
    )
    graph_repo = FalkorDBUserGraphRepository(graph_client)

    strategies: dict[str, AskStrategy] = {
        "normal": NormalStrategy(llm=generation_llm, max_tokens=settings.answer_max_tokens),
        "learning": LearningStrategy(llm=generation_llm, max_tokens=settings.answer_max_tokens),
        "concise": ConciseStrategy(llm=generation_llm, max_tokens=settings.answer_max_tokens),
    }

    return AskHandler(
        embedder=EmbeddingProviderFactory.from_settings(settings),
        chunks=PostgresChunkRepository(pool),
        classifier=KnowledgeClassifier(extractor=extractor, graph=graph_repo),
        post_processor=AnswerPostProcessor(extractor=extractor, graph=graph_repo),
        strategies=strategies,
        top_k=settings.top_k,
    )


IngestHandlerDep = Annotated[IngestDocumentHandler, Depends(get_ingest_handler)]
AskHandlerDep = Annotated[AskHandler, Depends(get_ask_handler)]
