from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import uuid4

from langchain_core.tools import BaseTool

from src.known_gap.application.services.tool_enabled_answerer import ToolEnabledAnswerer
from src.known_gap.application.strategies.base import StrategyContext
from src.known_gap.application.strategies.concise import ConciseStrategy
from src.known_gap.application.strategies.learning import LearningStrategy
from src.known_gap.application.strategies.normal import NormalStrategy
from src.known_gap.domain.models.concept import Concept, ConceptMention
from src.known_gap.domain.models.retrieved_chunk import RetrievedChunk


class _RecordingAnswerer(ToolEnabledAnswerer):
    """Bypass __init__ — we just need a stub that records prompts."""

    def __init__(self) -> None:  # noqa: D401 — intentionally skip super().__init__
        self.last_system: str | None = None
        self.last_user: str | None = None
        self.last_tools: Sequence[BaseTool] | None = None

    async def answer(  # type: ignore[override]
        self,
        system: str,
        user: str,
        tools: Sequence[BaseTool] | None = None,
    ) -> str:
        self.last_system = system
        self.last_user = user
        self.last_tools = tools
        return "ok"


def _chunk(content: str = "Python is interpreted.", filename: str = "intro.md") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        content=content,
        document_filename=filename,
        similarity_score=0.88,
    )


def _mention(name: str) -> ConceptMention:
    return ConceptMention(
        canonical_name=name.lower(),
        display_name=name,
        description="",
        domain=None,
    )


def _concept(name: str, score: int) -> Concept:
    now = datetime.now(UTC)
    return Concept(
        canonical_name=name.lower(),
        display_name=name,
        description="",
        domain=None,
        known_score=score,
        first_seen=now,
        last_seen=now,
    )


def _ctx(
    *,
    query: str = "Q?",
    retrieved: Sequence[RetrievedChunk] = (),
    known: Sequence[Concept] = (),
    unknown: Sequence[ConceptMention] = (),
    tools: Sequence[BaseTool] = (),
) -> StrategyContext:
    return StrategyContext(
        query=query,
        retrieved=retrieved,
        known=known,
        unknown=unknown,
        tools=tools,
    )


class TestNormalStrategy:
    async def test_passes_tools_through_and_excludes_known_list(self) -> None:
        answerer = _RecordingAnswerer()
        strategy = NormalStrategy(answerer)
        await strategy.answer(_ctx(query="What is Python?", retrieved=[_chunk()]))

        assert answerer.last_system is not None
        assert "ALREADY KNOWS" not in answerer.last_system
        assert answerer.last_user is not None
        assert "Python is interpreted." in answerer.last_user
        assert "What is Python?" in answerer.last_user

    async def test_no_context_prompt_when_retrieval_empty(self) -> None:
        answerer = _RecordingAnswerer()
        strategy = NormalStrategy(answerer)
        await strategy.answer(_ctx(query="q"))

        assert answerer.last_user is not None
        assert "No relevant context" in answerer.last_user


class TestLearningStrategy:
    async def test_learning_prompt_lists_known_concepts_with_scores(self) -> None:
        answerer = _RecordingAnswerer()
        strategy = LearningStrategy(answerer)
        await strategy.answer(
            _ctx(
                query="Explain memoisation.",
                retrieved=[_chunk()],
                known=[_concept("Recursion", 80), _concept("Base case", 60)],
                unknown=[_mention("Memoisation")],
            )
        )

        assert answerer.last_system is not None
        assert "LEARNING mode" in answerer.last_system
        assert "Recursion" in answerer.last_system
        assert "score=80" in answerer.last_system
        # The strategy must NOT instruct the model to insert tags itself —
        # cloze post-processing happens deterministically.
        assert "<cloze" not in answerer.last_system
        assert "<known" not in answerer.last_system

    async def test_falls_back_gracefully_with_no_known_concepts(self) -> None:
        answerer = _RecordingAnswerer()
        strategy = LearningStrategy(answerer)
        await strategy.answer(_ctx(query="q", retrieved=[_chunk()]))

        assert answerer.last_system is not None
        assert "(none)" in answerer.last_system


class TestConciseStrategy:
    async def test_instructs_skipping_known_concepts(self) -> None:
        answerer = _RecordingAnswerer()
        strategy = ConciseStrategy(answerer)
        await strategy.answer(
            _ctx(
                query="Explain decorators.",
                retrieved=[_chunk()],
                known=[_concept("Higher-order functions", 70)],
                unknown=[_mention("Decorators")],
            )
        )

        assert answerer.last_system is not None
        assert "CONCISE" in answerer.last_system
        assert "Higher-order functions" in answerer.last_system
        assert "Do NOT re-explain" in answerer.last_system
