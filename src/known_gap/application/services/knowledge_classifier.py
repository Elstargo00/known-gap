from uuid import UUID

from src.known_gap.application.services.concept_extractor import ConceptExtractor
from src.known_gap.domain.models.knowledge_classification import KnowledgeClassification
from src.known_gap.domain.repositories.user_graph_repository import UserGraphRepository


class KnowledgeClassifier:
    """Extracts concepts from text then partitions them into
    {known, unknown} against a user's persisted graph."""

    def __init__(self, extractor: ConceptExtractor, graph: UserGraphRepository) -> None:
        self._extractor = extractor
        self._graph = graph

    async def classify(self, user_id: UUID, text: str) -> KnowledgeClassification:
        mentions = await self._extractor.extract(text)
        if not mentions:
            return KnowledgeClassification(known=[], unknown=[])

        names = [m.canonical_name for m in mentions]
        known_names = await self._graph.find_known(user_id, names)

        known = [m for m in mentions if m.canonical_name in known_names]
        unknown = [m for m in mentions if m.canonical_name not in known_names]
        return KnowledgeClassification(known=known, unknown=unknown)
