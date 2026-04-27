from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from uuid import UUID

from src.known_gap.domain.models.concept import Concept, ConceptMention
from src.known_gap.domain.models.concept_neighborhood import ConceptNeighborhood
from src.known_gap.domain.models.concept_relation import ConceptRelation


class UserGraphRepository(ABC):
    """Port for the per-user knowledge graph.

    All queries and writes are scoped by `user_id` — implementations are
    free to materialize each user as a distinct database/graph. Scores
    are integers in [0, 100]; implementations clamp on writes.
    """

    @abstractmethod
    async def upsert_concepts(
        self,
        user_id: UUID,
        mentions: Sequence[ConceptMention],
        initial_score: int = 0,
    ) -> list[str]:
        """Idempotently create or refresh concept nodes.

        On CREATE the node is seeded with `known_score = initial_score`;
        on MATCH only `last_seen` and metadata fields are refreshed (the
        score is left to ScoreUpdater).

        Returns the canonical_names of nodes that were newly created in
        this call — used by the graph expander to know which seeds need
        densification.
        """

    @abstractmethod
    async def get_concepts(
        self,
        user_id: UUID,
        canonical_names: Sequence[str],
    ) -> list[Concept]:
        """Fetch persisted Concept objects (with known_score) for the
        given canonical names. Names not in the graph are simply omitted
        — the caller infers absence from the gap."""

    @abstractmethod
    async def adjust_scores(
        self,
        user_id: UUID,
        deltas: Mapping[str, int],
    ) -> None:
        """Apply integer deltas (positive or negative) to known_score.

        Implementations clamp the result to [0, 100]. Names not in the
        graph are silently ignored.
        """

    @abstractmethod
    async def add_relations(
        self,
        user_id: UUID,
        relations: Sequence[ConceptRelation],
    ) -> None:
        """Create typed relationship edges (idempotent — duplicate edges
        with the same (from, to, type) are merged, not duplicated).

        Both endpoints must already exist as Concept nodes; relations
        whose endpoints are missing are silently skipped (the caller is
        responsible for upserting nodes first).
        """

    @abstractmethod
    async def get_neighborhood(
        self,
        user_id: UUID,
        canonical_name: str,
        max_hops: int,
    ) -> ConceptNeighborhood | None:
        """Return the seed concept and all concepts reachable within
        `max_hops` of it, plus the connecting edges. Returns None if the
        seed is not in the graph."""

    @abstractmethod
    async def find_path(
        self,
        user_id: UUID,
        from_canonical: str,
        to_canonical: str,
        max_hops: int,
    ) -> list[Concept] | None:
        """Shortest path of concept nodes between two canonical names,
        or None if no path within `max_hops` exists."""
