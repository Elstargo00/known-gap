from collections.abc import Sequence
from uuid import UUID, uuid4

from src.known_gap.application.services.answer_post_processor import AnswerPostProcessor
from src.known_gap.application.services.concept_extractor import ConceptExtractor
from src.known_gap.domain.models.concept import ConceptMention
from src.known_gap.domain.repositories.user_graph_repository import UserGraphRepository
from src.known_gap.domain.services.llm_provider import LLMProvider


class _CapturingLLM(LLMProvider):
    def __init__(self, response: str) -> None:
        self._response = response
        self.last_user_prompt: str | None = None

    async def complete(self, system: str, user: str, max_tokens: int) -> str:
        self.last_user_prompt = user
        return self._response


class _CapturingGraph(UserGraphRepository):
    def __init__(self) -> None:
        self.upserted: list[tuple[UUID, Sequence[ConceptMention]]] = []

    async def find_known(self, user_id: UUID, canonical_names: Sequence[str]) -> set[str]:
        return set()

    async def upsert_concepts(self, user_id: UUID, mentions: Sequence[ConceptMention]) -> None:
        self.upserted.append((user_id, mentions))


class TestAnswerPostProcessor:
    async def test_strips_known_tags_before_extraction(self) -> None:
        llm = _CapturingLLM(response='{"concepts": [{"name": "X"}]}')
        graph = _CapturingGraph()
        processor = AnswerPostProcessor(ConceptExtractor(llm), graph)

        answer = 'Recursion uses <known concept="base-case">a base case</known> to terminate.'
        await processor.update_graph(uuid4(), answer)

        assert llm.last_user_prompt is not None
        assert "<known" not in llm.last_user_prompt
        assert "</known>" not in llm.last_user_prompt
        assert "a base case" in llm.last_user_prompt

    async def test_upserts_extracted_concepts_for_user(self) -> None:
        llm = _CapturingLLM(
            response=('{"concepts": [{"name": "Recursion"},{"name": "Base case"}]}')
        )
        graph = _CapturingGraph()
        processor = AnswerPostProcessor(ConceptExtractor(llm), graph)

        user_id = uuid4()
        await processor.update_graph(user_id, "any answer")

        assert len(graph.upserted) == 1
        recorded_user, mentions = graph.upserted[0]
        assert recorded_user == user_id
        assert {m.canonical_name for m in mentions} == {"recursion", "base case"}

    async def test_noop_on_empty_answer(self) -> None:
        llm = _CapturingLLM(response="should not be called")
        graph = _CapturingGraph()
        processor = AnswerPostProcessor(ConceptExtractor(llm), graph)

        await processor.update_graph(uuid4(), "   ")

        assert llm.last_user_prompt is None
        assert graph.upserted == []
