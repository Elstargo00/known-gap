from collections.abc import Sequence
from uuid import UUID

from src.known_gap.application.services.concept_extractor import ConceptExtractor
from src.known_gap.domain.models.concept import ConceptMention
from src.known_gap.domain.models.knowledge_classification import KnowledgeClassification
from src.known_gap.domain.repositories.user_graph_repository import UserGraphRepository


class KnowledgeClassifier:
    """Extracts concepts from text then partitions them against a user's
    persisted graph using `known_score` and a configurable threshold.

    A concept is "known" when:
      - it exists as a node in the user's graph, AND
      - its `known_score` is strictly above `threshold`.

    Anything else is "unknown" — including concepts that exist in the
    graph but haven't yet been reinforced past the threshold (i.e. seen
    once but not yet "owned"). This is what enables the cloze quiz
    behaviour: the user sees the concept until they've engaged with it
    enough times to push the score over the bar, and only then does the
    UI start hiding it.
    """

    def __init__(
        self,
        extractor: ConceptExtractor,
        graph: UserGraphRepository,
        threshold: int,
    ) -> None:
        self._extractor = extractor
        self._graph = graph
        self._threshold = threshold

    async def extract(self, text: str) -> list[ConceptMention]:
        return await self._extractor.extract(text)

    async def classify_mentions(
        self,
        user_id: UUID,
        mentions: Sequence[ConceptMention],
    ) -> KnowledgeClassification:
        if not mentions:
            return KnowledgeClassification(known=[], unknown=[])

        names = [m.canonical_name for m in mentions]
        persisted = await self._graph.get_concepts(user_id, names)
        score_by_name = {c.canonical_name: c for c in persisted}

        known = [
            score_by_name[m.canonical_name]
            for m in mentions
            if m.canonical_name in score_by_name
            and score_by_name[m.canonical_name].known_score > self._threshold
        ]
        known_names = {c.canonical_name for c in known}
        unknown = [m for m in mentions if m.canonical_name not in known_names]
        return KnowledgeClassification(known=known, unknown=unknown)

    async def classify(self, user_id: UUID, text: str) -> KnowledgeClassification:
        """Convenience: extract + classify in one call."""
        mentions = await self.extract(text)
        return await self.classify_mentions(user_id, mentions)
