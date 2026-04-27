from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from src.known_gap.domain.models.concept import ConceptMention
from src.known_gap.domain.models.concept_relation import ConceptRelation


@dataclass(frozen=True)
class ExpertProposal:
    """Output of a GraphExpert call: optional new concept nodes plus the
    typed relations connecting seeds (and possibly the new nodes)."""

    new_concepts: list[ConceptMention]
    relations: list[ConceptRelation]


class GraphExpert(ABC):
    """Port: an LLM-backed expert that proposes graph edges (and possibly
    new concept nodes) around a set of seed concepts in order to densify
    a user's knowledge graph.

    Implementations are expected to use a high-capability model (Claude
    Opus by default) since they bring "world knowledge" to bear — the
    expert reasons about which concepts plausibly relate and how, beyond
    what's literally in the source text.
    """

    @abstractmethod
    async def propose(
        self,
        seed_concepts: Sequence[ConceptMention],
        existing_neighbors: Sequence[ConceptMention],
        max_per_seed: int,
    ) -> ExpertProposal:
        """Propose up to `max_per_seed` outgoing relations for each seed.

        The expert may emit new concepts (which the caller will upsert
        with score=0) and edges between any combination of seeds, new
        concepts, and existing neighbors.
        """
