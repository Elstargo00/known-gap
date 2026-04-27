from datetime import UTC, datetime
from uuid import uuid4

from src.known_gap.application.services.concept_extractor import ConceptExtractor
from src.known_gap.application.services.knowledge_classifier import KnowledgeClassifier
from src.known_gap.domain.models.concept import Concept
from tests.unit._fakes import FakeLLM, FakeUserGraphRepository


def _seeded_graph(user_id, scores: dict[str, int]) -> FakeUserGraphRepository:
    graph = FakeUserGraphRepository()
    bucket = graph._user_concepts(user_id)  # noqa: SLF001 — test seam
    now = datetime.now(UTC)
    for name, score in scores.items():
        bucket[name] = Concept(
            canonical_name=name,
            display_name=name.title(),
            description="",
            domain=None,
            known_score=score,
            first_seen=now,
            last_seen=now,
        )
    return graph


class TestKnowledgeClassifier:
    async def test_partitions_concepts_by_score_threshold(self) -> None:
        user_id = uuid4()
        # recursion is firmly known (60), base case is below threshold (40),
        # trampolining is not in the graph at all.
        graph = _seeded_graph(user_id, {"recursion": 60, "base case": 40})
        extractor = ConceptExtractor(
            FakeLLM(
                '{"concepts": ['
                '{"name": "Recursion"},{"name": "Base Case"},{"name": "Trampolining"}]}'
            )
        )
        classifier = KnowledgeClassifier(extractor=extractor, graph=graph, threshold=50)

        result = await classifier.classify(user_id, "explain recursion")

        assert {c.canonical_name for c in result.known} == {"recursion"}
        assert {m.canonical_name for m in result.unknown} == {"base case", "trampolining"}

    async def test_known_concepts_carry_known_score(self) -> None:
        user_id = uuid4()
        graph = _seeded_graph(user_id, {"recursion": 80})
        extractor = ConceptExtractor(FakeLLM('{"concepts": [{"name": "Recursion"}]}'))
        classifier = KnowledgeClassifier(extractor=extractor, graph=graph, threshold=50)

        result = await classifier.classify(user_id, "explain recursion")

        assert len(result.known) == 1
        assert result.known[0].known_score == 80

    async def test_empty_extraction_yields_empty_classification(self) -> None:
        graph = FakeUserGraphRepository()
        extractor = ConceptExtractor(FakeLLM('{"concepts": []}'))
        classifier = KnowledgeClassifier(extractor=extractor, graph=graph, threshold=50)

        result = await classifier.classify(uuid4(), "gibberish")

        assert result.known == []
        assert result.unknown == []
