from uuid import uuid4

from src.known_gap.application.services.answer_post_processor import AnswerPostProcessor
from src.known_gap.application.services.concept_extractor import ConceptExtractor
from tests.unit._fakes import FakeLLM, FakeUserGraphRepository


class TestAnswerPostProcessor:
    async def test_strips_cloze_tags_before_extraction(self) -> None:
        llm = FakeLLM('{"concepts": [{"name": "X"}]}')
        graph = FakeUserGraphRepository()
        processor = AnswerPostProcessor(ConceptExtractor(llm), graph)

        answer = 'Recursion uses <cloze concept="base-case">a base case</cloze> to terminate.'
        await processor.extract_and_persist(uuid4(), answer)

        assert llm.calls, "extractor should have been called"
        _, prompt, _ = llm.calls[0]
        assert "<cloze" not in prompt
        assert "</cloze>" not in prompt
        assert "a base case" in prompt

    async def test_upserts_extracted_concepts_for_user(self) -> None:
        llm = FakeLLM('{"concepts": [{"name": "Recursion"},{"name": "Base case"}]}')
        graph = FakeUserGraphRepository()
        processor = AnswerPostProcessor(ConceptExtractor(llm), graph)

        user_id = uuid4()
        mentions, newly_created = await processor.extract_and_persist(user_id, "any answer")

        assert {m.canonical_name for m in mentions} == {"recursion", "base case"}
        assert set(newly_created) == {"recursion", "base case"}
        assert len(graph.upsert_calls) == 1
        recorded_user, recorded_mentions, _ = graph.upsert_calls[0]
        assert recorded_user == user_id
        assert {m.canonical_name for m in recorded_mentions} == {"recursion", "base case"}

    async def test_noop_on_empty_answer(self) -> None:
        llm = FakeLLM("should not be called")
        graph = FakeUserGraphRepository()
        processor = AnswerPostProcessor(ConceptExtractor(llm), graph)

        mentions, newly = await processor.extract_and_persist(uuid4(), "   ")

        assert mentions == []
        assert newly == []
        assert llm.calls == []
        assert graph.upsert_calls == []
