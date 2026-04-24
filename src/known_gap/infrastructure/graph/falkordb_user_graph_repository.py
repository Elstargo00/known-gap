from collections.abc import Sequence
from uuid import UUID

from falkordb.asyncio import FalkorDB

from src.known_gap.domain.models.concept import ConceptMention
from src.known_gap.domain.repositories.user_graph_repository import UserGraphRepository


class FalkorDBUserGraphRepository(UserGraphRepository):
    SEED_CONFIDENCE = 0.3
    CONFIDENCE_BUMP = 0.1
    CONFIDENCE_MAX = 1.0

    def __init__(self, client: FalkorDB) -> None:
        self._client = client

    def _graph_name(self, user_id: UUID) -> str:
        return f"user_{user_id.hex}"

    async def find_known(
        self,
        user_id: UUID,
        canonical_names: Sequence[str],
    ) -> set[str]:
        names = list(canonical_names)
        if not names:
            return set()
        graph = self._client.select_graph(self._graph_name(user_id))
        result = await graph.query(
            """
            MATCH (c:Concept)
            WHERE c.canonical_name IN $names
            RETURN c.canonical_name
            """,
            {"names": names},
        )
        return {row[0] for row in result.result_set}

    async def upsert_concepts(
        self,
        user_id: UUID,
        mentions: Sequence[ConceptMention],
    ) -> None:
        if not mentions:
            return
        graph = self._client.select_graph(self._graph_name(user_id))
        payload = [
            {
                "canonical_name": m.canonical_name,
                "display_name": m.display_name,
                "description": m.description,
                "domain": m.domain,
            }
            for m in mentions
        ]
        await graph.query(
            """
            UNWIND $mentions AS m
            MERGE (c:Concept {canonical_name: m.canonical_name})
            ON CREATE SET
                c.display_name      = m.display_name,
                c.description       = m.description,
                c.domain            = m.domain,
                c.confidence        = $seed,
                c.times_encountered = 1,
                c.first_seen        = timestamp(),
                c.last_seen         = timestamp()
            ON MATCH SET
                c.confidence = CASE
                    WHEN c.confidence + $bump > $max_conf THEN $max_conf
                    ELSE c.confidence + $bump
                END,
                c.times_encountered = c.times_encountered + 1,
                c.last_seen         = timestamp()
            """,
            {
                "mentions": payload,
                "seed": self.SEED_CONFIDENCE,
                "bump": self.CONFIDENCE_BUMP,
                "max_conf": self.CONFIDENCE_MAX,
            },
        )
