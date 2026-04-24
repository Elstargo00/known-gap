from uuid import uuid4

from src.known_gap.application.strategies.base import StrategyContext
from src.known_gap.application.strategies.concise import ConciseStrategy
from src.known_gap.application.strategies.learning import LearningStrategy
from src.known_gap.application.strategies.normal import NormalStrategy
from src.known_gap.domain.models.concept import ConceptMention
from src.known_gap.domain.models.retrieved_chunk import RetrievedChunk
from src.known_gap.domain.services.llm_provider import LLMProvider


class _RecordingLLM(LLMProvider):
    def __init__(self) -> None:
        self.last_system: str | None = None
        self.last_user: str | None = None

    async def complete(self, system: str, user: str, max_tokens: int) -> str:
        self.last_system = system
        self.last_user = user
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


class TestNormalStrategy:
    async def test_ignores_known_concepts_in_prompt(self) -> None:
        llm = _RecordingLLM()
        strategy = NormalStrategy(llm, max_tokens=512)
        context = StrategyContext(
            query="What is Python?",
            retrieved=[_chunk()],
            known=[_mention("Recursion")],
            unknown=[],
        )
        await strategy.answer(context)

        assert llm.last_system is not None
        assert "ALREADY KNOWS" not in llm.last_system
        assert llm.last_user is not None
        assert "Python is interpreted." in llm.last_user
        assert "What is Python?" in llm.last_user

    async def test_uses_no_context_prompt_when_retrieval_empty(self) -> None:
        llm = _RecordingLLM()
        strategy = NormalStrategy(llm, max_tokens=512)
        context = StrategyContext(query="q", retrieved=[], known=[], unknown=[])
        await strategy.answer(context)

        assert llm.last_user is not None
        assert "No relevant context" in llm.last_user


class TestLearningStrategy:
    async def test_injects_known_list_and_requests_inline_tags(self) -> None:
        llm = _RecordingLLM()
        strategy = LearningStrategy(llm, max_tokens=512)
        context = StrategyContext(
            query="Explain memoisation.",
            retrieved=[_chunk()],
            known=[_mention("Recursion"), _mention("Base case")],
            unknown=[_mention("Memoisation")],
        )
        await strategy.answer(context)

        assert llm.last_system is not None
        assert "LEARNING mode" in llm.last_system
        assert "Recursion" in llm.last_system
        assert "Base case" in llm.last_system
        assert "<known concept=" in llm.last_system

    async def test_marks_no_known_concepts_gracefully(self) -> None:
        llm = _RecordingLLM()
        strategy = LearningStrategy(llm, max_tokens=512)
        context = StrategyContext(query="q", retrieved=[_chunk()], known=[], unknown=[])
        await strategy.answer(context)
        assert llm.last_system is not None
        assert "(none)" in llm.last_system


class TestConciseStrategy:
    async def test_instructs_skipping_known_concepts(self) -> None:
        llm = _RecordingLLM()
        strategy = ConciseStrategy(llm, max_tokens=512)
        context = StrategyContext(
            query="Explain decorators.",
            retrieved=[_chunk()],
            known=[_mention("Higher-order functions")],
            unknown=[_mention("Decorators")],
        )
        await strategy.answer(context)

        assert llm.last_system is not None
        assert "CONCISE" in llm.last_system
        assert "Higher-order functions" in llm.last_system
        assert "Do NOT re-explain" in llm.last_system
