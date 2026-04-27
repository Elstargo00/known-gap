from typing import Annotated
from uuid import UUID

from anthropic import AsyncAnthropic
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.known_gap.application.services.answer_post_processor import AnswerPostProcessor
from src.known_gap.application.services.chunker import Chunker
from src.known_gap.application.services.cloze_processor import ClozeProcessor
from src.known_gap.application.services.concept_extractor import ConceptExtractor
from src.known_gap.application.services.graph_expander import GraphExpander
from src.known_gap.application.services.knowledge_classifier import KnowledgeClassifier
from src.known_gap.application.services.score_updater import ScoreUpdater
from src.known_gap.application.services.tool_enabled_answerer import ToolEnabledAnswerer
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
from src.known_gap.infrastructure.llm.anthropic_graph_expert import AnthropicGraphExpert
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

bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user_id(
    verifier: JWTVerifierDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)] = None,
) -> UUID:
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid Authorization header",
        )
    try:
        return verifier.verify(credentials.credentials)
    except AppException as e:
        raise HTTPException(status_code=e.http_status_code, detail=e.message) from e


CurrentUserId = Annotated[UUID, Depends(get_current_user_id)]


def _build_graph_expander(
    settings: Settings, graph_repo: FalkorDBUserGraphRepository
) -> GraphExpander:
    expert_llm = LLMProviderFactory.for_graph_expert(settings)
    expert = AnthropicGraphExpert(
        llm=expert_llm,
        max_tokens=settings.graph_expert_max_tokens,
    )
    return GraphExpander(
        graph=graph_repo,
        expert=expert,
        max_per_seed=settings.graph_expert_degree,
    )


def get_ingest_handler(
    request: Request,
    settings: SettingsDep,
) -> IngestDocumentHandler:
    pool = request.app.state.pool
    graph_client = request.app.state.graph_client
    graph_repo = FalkorDBUserGraphRepository(graph_client)
    concept_llm = LLMProviderFactory.for_concept_extraction(settings)
    extractor = ConceptExtractor(
        llm=concept_llm,
        max_tokens=settings.concept_extraction_max_tokens,
    )
    return IngestDocumentHandler(
        parser=DispatchingDocumentParser(),
        chunker=Chunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        ),
        embedder=EmbeddingProviderFactory.from_settings(settings),
        documents=PostgresDocumentRepository(pool),
        chunks=PostgresChunkRepository(pool),
        concept_extractor=extractor,
        graph=graph_repo,
        graph_expander=_build_graph_expander(settings, graph_repo),
        initial_score=settings.known_score_initial,
    )


def get_ask_handler(
    request: Request,
    settings: SettingsDep,
) -> AskHandler:
    pool = request.app.state.pool
    graph_client = request.app.state.graph_client
    graph_repo = FalkorDBUserGraphRepository(graph_client)

    if not settings.anthropic_api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY is required")
    anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    fallback_llm = LLMProviderFactory.from_settings(settings)
    answerer = ToolEnabledAnswerer(
        client=anthropic_client,
        model=settings.llm_primary_model,
        max_tokens=settings.answer_max_tokens,
        max_iterations=settings.tool_max_iterations,
        fallback_llm=fallback_llm,
    )

    concept_llm = LLMProviderFactory.for_concept_extraction(settings)
    extractor = ConceptExtractor(
        llm=concept_llm,
        max_tokens=settings.concept_extraction_max_tokens,
    )

    classifier = KnowledgeClassifier(
        extractor=extractor,
        graph=graph_repo,
        threshold=settings.known_score_threshold,
    )
    post_processor = AnswerPostProcessor(
        extractor=extractor,
        graph=graph_repo,
        initial_score=settings.known_score_initial,
    )
    score_updater = ScoreUpdater(
        graph=graph_repo,
        threshold=settings.known_score_threshold,
        increment=settings.known_score_increment,
        decrement=settings.known_score_decrement,
    )
    cloze_processor = ClozeProcessor(threshold=settings.known_score_threshold)
    graph_expander = _build_graph_expander(settings, graph_repo)

    strategies: dict[str, AskStrategy] = {
        "normal": NormalStrategy(answerer=answerer),
        "learning": LearningStrategy(answerer=answerer),
        "concise": ConciseStrategy(answerer=answerer),
    }

    return AskHandler(
        embedder=EmbeddingProviderFactory.from_settings(settings),
        chunks=PostgresChunkRepository(pool),
        graph=graph_repo,
        classifier=classifier,
        post_processor=post_processor,
        score_updater=score_updater,
        cloze_processor=cloze_processor,
        graph_expander=graph_expander,
        strategies=strategies,
        top_k=settings.top_k,
        graph_tool_max_hops=settings.concept_neighborhood_max_hops,
    )


IngestHandlerDep = Annotated[IngestDocumentHandler, Depends(get_ingest_handler)]
AskHandlerDep = Annotated[AskHandler, Depends(get_ask_handler)]
