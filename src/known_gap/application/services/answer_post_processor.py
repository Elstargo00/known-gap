from uuid import UUID

from src.known_gap.application.services.cloze_processor import ClozeProcessor
from src.known_gap.application.services.concept_extractor import ConceptExtractor
from src.known_gap.domain.models.concept import ConceptMention
from src.known_gap.domain.repositories.user_graph_repository import UserGraphRepository


class AnswerPostProcessor:
    """Post-answer side effects:

      1. Strip presentation tags (e.g. <cloze>) from the answer.
      2. Extract concepts that actually appeared in the answer.
      3. Upsert them into the user's graph with `known_score = 0` if new.

    This service is intentionally focused on *extraction + node upsert*
    only. Score adjustments are applied separately by `ScoreUpdater` and
    edge densification is fired off by `GraphExpander` — both consume the
    list of canonical names returned here.
    """

    def __init__(
        self,
        extractor: ConceptExtractor,
        graph: UserGraphRepository,
        initial_score: int = 0,
    ) -> None:
        self._extractor = extractor
        self._graph = graph
        self._initial_score = initial_score

    async def extract_and_persist(
        self,
        user_id: UUID,
        answer_text: str,
    ) -> tuple[list[ConceptMention], list[str]]:
        """Returns (all_extracted_mentions, newly_created_canonicals)."""
        cleaned = ClozeProcessor.strip_tags(answer_text)
        if not cleaned.strip():
            return [], []
        mentions = await self._extractor.extract(cleaned)
        if not mentions:
            return [], []
        newly_created = await self._graph.upsert_concepts(
            user_id, mentions, initial_score=self._initial_score
        )
        return mentions, newly_created
