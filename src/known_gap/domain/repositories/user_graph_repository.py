from abc import ABC, abstractmethod
from collections.abc import Sequence
from uuid import UUID

from src.known_gap.domain.models.concept import ConceptMention


class UserGraphRepository(ABC):
    @abstractmethod
    async def find_known(
        self,
        user_id: UUID,
        canonical_names: Sequence[str],
    ) -> set[str]:
        """Return the subset of canonical_names that already exist as
        Concept nodes in the given user's knowledge graph."""

    @abstractmethod
    async def upsert_concepts(
        self,
        user_id: UUID,
        mentions: Sequence[ConceptMention],
    ) -> None:
        """Create a Concept node for each new mention; for existing
        concepts bump confidence, increment times_encountered, and
        refresh last_seen."""
