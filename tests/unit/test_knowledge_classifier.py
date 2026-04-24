from collections.abc import Sequence
from uuid import UUID, uuid4

from src.known_gap.application.services.concept_extractor import ConceptExtractor
from src.known_gap.application.services.knowledge_classifier import KnowledgeClassifier
from src.known_gap.domain.models.concept import ConceptMention
from src.known_gap.domain.repositories.user_graph_repository import UserGraphRepository
from src.known_gap.domain.services.llm_provider import LLMProvider


class _StubLLM(LLMProvider):
    def __init__(self, response: str) -> None:
        self._response = response

    async def complete(self, system: str, user: str, max_tokens: int) -> str:
        return self._response


class _StubGraph(UserGraphRepository):
    def __init__(self, known: set[str]) -> None:
        self._known = known
        self.upserted: list[Sequence[ConceptMention]] = []

    async def find_known(self, user_id: UUID, canonical_names: Sequence[str]) -> set[str]:
        return {name for name in canonical_names if name in self._known}

    async def upsert_concepts(self, user_id: UUID, mentions: Sequence[ConceptMention]) -> None:
        self.upserted.append(mentions)


def _extractor(response: str) -> ConceptExtractor:
    return ConceptExtractor(_StubLLM(response))


class TestKnowledgeClassifier:
    async def test_partitions_concepts_by_graph_presence(self) -> None:
        extractor = _extractor(
            '{"concepts": [{"name": "Recursion"},{"name": "Base Case"},{"name": "Trampolining"}]}'
        )
        graph = _StubGraph(known={"recursion", "base case"})
        classifier = KnowledgeClassifier(extractor, graph)

        result = await classifier.classify(uuid4(), "explain recursion")
        known_names = [m.canonical_name for m in result.known]
        unknown_names = [m.canonical_name for m in result.unknown]

        assert set(known_names) == {"recursion", "base case"}
        assert unknown_names == ["trampolining"]

    async def test_empty_extraction_yields_empty_classification(self) -> None:
        classifier = KnowledgeClassifier(
            extractor=_extractor('{"concepts": []}'),
            graph=_StubGraph(known=set()),
        )
        result = await classifier.classify(uuid4(), "gibberish")
        assert result.known == []
        assert result.unknown == []
