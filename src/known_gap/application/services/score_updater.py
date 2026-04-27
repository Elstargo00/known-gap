from collections.abc import Iterable, Sequence
from uuid import UUID

from src.known_gap.domain.models.concept import Concept, ConceptMention
from src.known_gap.domain.repositories.user_graph_repository import UserGraphRepository


class ScoreUpdater:
    """Encapsulates how a user's `known_score` moves after an exchange.

    Option 2 rules (per design doc):
      - A concept that appears in the user's question AND is currently
        "known" (score > threshold) is *re-asked* — they probably forgot.
        Apply -decrement.
      - A concept that appears in the answer (and was NOT in the
        question) is *introduced/reinforced* — apply +increment.
      - All scores clamp to [0, 100] (delegated to the repository).

    Concepts that are in BOTH question and answer get neither bonus nor
    penalty: they are "the topic", not new exposure or forgotten ground.
    """

    def __init__(
        self,
        graph: UserGraphRepository,
        threshold: int,
        increment: int,
        decrement: int,
    ) -> None:
        self._graph = graph
        self._threshold = threshold
        self._increment = increment
        self._decrement = decrement

    async def apply(
        self,
        user_id: UUID,
        question_concepts: Sequence[ConceptMention],
        answer_concepts: Sequence[ConceptMention],
        known_in_question: Sequence[Concept],
    ) -> dict[str, int]:
        """Return the deltas that were applied (post-clamp not reflected
        — the caller can re-fetch if needed)."""
        question_names = {c.canonical_name for c in question_concepts}
        answer_names = {c.canonical_name for c in answer_concepts}

        deltas: dict[str, int] = {}

        # Re-asked penalty: known concepts that the user is querying again.
        for concept in known_in_question:
            if concept.known_score > self._threshold:
                deltas[concept.canonical_name] = (
                    deltas.get(concept.canonical_name, 0) - self._decrement
                )

        # Reinforcement bonus: concepts introduced via the answer only.
        for name in answer_names - question_names:
            deltas[name] = deltas.get(name, 0) + self._increment

        if deltas:
            await self._graph.adjust_scores(user_id, deltas)
        return deltas

    @staticmethod
    def coalesce(*sources: Iterable[ConceptMention]) -> list[ConceptMention]:
        """Helper to merge multiple mention iterables, deduping on
        canonical_name (first-seen wins)."""
        seen: set[str] = set()
        out: list[ConceptMention] = []
        for source in sources:
            for mention in source:
                if mention.canonical_name in seen:
                    continue
                seen.add(mention.canonical_name)
                out.append(mention)
        return out
