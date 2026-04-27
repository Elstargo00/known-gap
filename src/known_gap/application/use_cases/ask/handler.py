import asyncio
import logging
from collections.abc import Mapping

from src.known_gap.application.services.answer_post_processor import AnswerPostProcessor
from src.known_gap.application.services.cloze_processor import ClozeProcessor
from src.known_gap.application.services.graph_expander import GraphExpander
from src.known_gap.application.services.knowledge_classifier import KnowledgeClassifier
from src.known_gap.application.services.score_updater import ScoreUpdater
from src.known_gap.application.strategies.base import AskStrategy, StrategyContext
from src.known_gap.application.tools.graph_tools import build_graph_tools
from src.known_gap.application.use_cases.ask.command import AskCommand
from src.known_gap.application.use_cases.ask.result import AskConcept, AskResult, AskSource
from src.known_gap.domain.repositories.chunk_repository import ChunkRepository
from src.known_gap.domain.repositories.user_graph_repository import UserGraphRepository
from src.known_gap.domain.services.embedding_provider import EmbeddingProvider
from src.known_gap.shared.exceptions.base import PermanentException

logger = logging.getLogger(__name__)

PREVIEW_CHARS = 200


class AskHandler:
    """Orchestrates the full Known Gap /ask flow.

    Steps:
      1. Embed the question + extract+classify question concepts in
         parallel.
      2. Retrieve the top-k chunks via pgvector.
      3. Build per-request graph tools and dispatch to the chosen
         strategy. The strategy's answerer can call the graph tools at
         will (true LLM-driven tool calling).
      4. Cloze-process the answer for `learning` mode (mask known
         concepts above threshold).
      5. Persist answer concepts (score=0 if new), apply score deltas
         (-decrement on re-asked-known, +increment on newly introduced).
      6. Fire-and-forget background graph expansion around all touched
         concepts so future answers see a denser graph.
    """

    def __init__(
        self,
        embedder: EmbeddingProvider,
        chunks: ChunkRepository,
        graph_repo: UserGraphRepository,
        classifier: KnowledgeClassifier,
        post_processor: AnswerPostProcessor,
        score_updater: ScoreUpdater,
        cloze_processor: ClozeProcessor,
        graph_expander: GraphExpander,
        strategies: Mapping[str, AskStrategy],
        top_k: int,
        graph_tool_max_hops: int,
    ) -> None:
        self._embedder = embedder
        self._chunks = chunks
        self._graph = graph_repo
        self._classifier = classifier
        self._post_processor = post_processor
        self._score_updater = score_updater
        self._cloze_processor = cloze_processor
        self._graph_expander = graph_expander
        self._strategies = strategies
        self._top_k = top_k
        self._graph_tool_max_hops = graph_tool_max_hops

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

        # 1. Embed + extract concepts from question in parallel; then
        # classify the extracted mentions against the user's graph.
        query_embedding, question_mentions = await asyncio.gather(
            self._embedder.embed_query(command.query),
            self._classifier.extract(command.query),
        )
        classification = await self._classifier.classify_mentions(
            command.user_id, question_mentions
        )

        # 2. Retrieve.
        retrieved = await self._chunks.search_similar(query_embedding, self._top_k)

        # 3. Build graph tools + dispatch to strategy.
        tools = build_graph_tools(self._graph, command.user_id, max_hops=self._graph_tool_max_hops)
        context = StrategyContext(
            query=command.query,
            retrieved=retrieved,
            known=classification.known,
            unknown=classification.unknown,
            tools=tools,
        )
        raw_answer = await strategy.answer(context)

        # 4. Persist answer concepts + update scores synchronously so the
        # *next* /ask call already reflects the exchange.
        answer_mentions, _newly_created = await self._post_processor.extract_and_persist(
            command.user_id, raw_answer
        )

        await self._score_updater.apply(
            user_id=command.user_id,
            question_concepts=question_mentions,
            answer_concepts=answer_mentions,
            known_in_question=classification.known,
        )

        # 5. Cloze-process for learning mode (after persistence so scores
        # are fresh when the cloze decision is made on the *current*
        # turn — but we still gate on the pre-call classification for
        # stability of the user-visible answer).
        cloze_canonicals: list[str] = []
        final_answer = raw_answer
        if command.mode == "learning":
            final_answer, cloze_canonicals = self._cloze_processor.process(
                raw_answer, classification.known
            )

        # 6. Fire-and-forget graph expansion around every concept touched
        # this turn. Background task; never raises.
        touched = list(
            {
                *(m.canonical_name for m in question_mentions),
                *(m.canonical_name for m in answer_mentions),
            }
        )
        if touched:
            asyncio.create_task(self._graph_expander.expand(command.user_id, touched))

        return AskResult(
            answer=final_answer,
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
                AskConcept(
                    canonical_name=c.canonical_name,
                    display_name=c.display_name,
                    known_score=c.known_score,
                )
                for c in classification.known
            ],
            unknown_concepts=[
                AskConcept(canonical_name=c.canonical_name, display_name=c.display_name)
                for c in classification.unknown
            ],
            cloze_concepts=cloze_canonicals,
        )
