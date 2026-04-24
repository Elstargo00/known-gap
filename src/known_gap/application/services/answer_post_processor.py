import re
from uuid import UUID

from src.known_gap.application.services.concept_extractor import ConceptExtractor
from src.known_gap.domain.repositories.user_graph_repository import UserGraphRepository

_KNOWN_TAG_PATTERN = re.compile(r"<known[^>]*>(.*?)</known>", re.DOTALL)


class AnswerPostProcessor:
    """Reads a generated answer (possibly with <known> tags from learning
    mode), extracts concepts, and upserts them into the user's graph so
    future queries reflect the latest exposure."""

    def __init__(self, extractor: ConceptExtractor, graph: UserGraphRepository) -> None:
        self._extractor = extractor
        self._graph = graph

    async def update_graph(self, user_id: UUID, answer_text: str) -> None:
        cleaned = _KNOWN_TAG_PATTERN.sub(r"\1", answer_text)
        if not cleaned.strip():
            return
        mentions = await self._extractor.extract(cleaned)
        await self._graph.upsert_concepts(user_id, mentions)
