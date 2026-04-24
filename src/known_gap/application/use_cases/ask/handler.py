import asyncio
from collections.abc import Mapping

from src.known_gap.application.services.answer_post_processor import AnswerPostProcessor
from src.known_gap.application.services.knowledge_classifier import KnowledgeClassifier
from src.known_gap.application.strategies.base import AskStrategy, StrategyContext
from src.known_gap.application.use_cases.ask.command import AskCommand
from src.known_gap.application.use_cases.ask.result import AskConcept, AskResult, AskSource
from src.known_gap.domain.models.knowledge_classification import KnowledgeClassification
from src.known_gap.domain.repositories.chunk_repository import ChunkRepository
from src.known_gap.domain.services.embedding_provider import EmbeddingProvider
from src.known_gap.shared.exceptions.base import PermanentException

PREVIEW_CHARS = 200


class AskHandler:
    """Orchestrates the full Known Gap /ask flow:

    - normal mode: pure RAG — embed, retrieve, generate, then post-update
      the user's graph from the answer.
    - learning / concise modes: additionally extract concepts from the
      query and classify them against the user's graph so the selected
      strategy can adapt its prompt (tag known spans or skip explanation).
    """

    def __init__(
        self,
        embedder: EmbeddingProvider,
        chunks: ChunkRepository,
        classifier: KnowledgeClassifier,
        post_processor: AnswerPostProcessor,
        strategies: Mapping[str, AskStrategy],
        top_k: int,
    ) -> None:
        self._embedder = embedder
        self._chunks = chunks
        self._classifier = classifier
        self._post_processor = post_processor
        self._strategies = strategies
        self._top_k = top_k

    async def execute(self, command: AskCommand) -> AskResult:
        strategy = self._strategies.get(command.mode)
        if strategy is None:
            raise PermanentException(
                message=f"Unsupported mode: {command.mode}",
                error_code="MODE_NOT_IMPLEMENTED",
                details={
                    "requested_mode": command.mode,
                    "supported": sorted(self._strategies.keys()),
                },
            )

        classification = KnowledgeClassification(known=[], unknown=[])
        if command.mode == "normal":
            query_embedding = await self._embedder.embed_query(command.query)
        else:
            query_embedding, classification = await asyncio.gather(
                self._embedder.embed_query(command.query),
                self._classifier.classify(command.user_id, command.query),
            )

        retrieved = await self._chunks.search_similar(query_embedding, self._top_k)

        context = StrategyContext(
            query=command.query,
            retrieved=retrieved,
            known=classification.known,
            unknown=classification.unknown,
        )
        answer = await strategy.answer(context)

        await self._post_processor.update_graph(command.user_id, answer)

        return AskResult(
            answer=answer,
            sources=[
                AskSource(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    filename=chunk.document_filename,
                    content_preview=chunk.content[:PREVIEW_CHARS],
                    similarity_score=chunk.similarity_score,
                )
                for chunk in retrieved
            ],
            mode=command.mode,
            known_concepts=[
                AskConcept(canonical_name=c.canonical_name, display_name=c.display_name)
                for c in classification.known
            ],
            unknown_concepts=[
                AskConcept(canonical_name=c.canonical_name, display_name=c.display_name)
                for c in classification.unknown
            ],
        )
