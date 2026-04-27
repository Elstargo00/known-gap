import logging
from collections.abc import Sequence
from uuid import UUID

from src.known_gap.domain.models.concept import Concept, ConceptMention
from src.known_gap.domain.repositories.user_graph_repository import UserGraphRepository
from src.known_gap.domain.services.graph_expert import GraphExpert

logger = logging.getLogger(__name__)


class GraphExpander:
    """Densifies a user's knowledge graph by asking a GraphExpert for
    new typed edges around a set of seed concepts.

    The expander is designed to run as a background task fired off by
    /ingest and /ask handlers — it never raises into the request path:
    any failure is logged and swallowed.
    """

    def __init__(
        self,
        graph: UserGraphRepository,
        expert: GraphExpert,
        max_per_seed: int,
    ) -> None:
        self._graph = graph
        self._expert = expert
        self._max_per_seed = max_per_seed

    async def expand(
        self,
        user_id: UUID,
        seed_canonicals: Sequence[str],
    ) -> None:
        if not seed_canonicals or self._max_per_seed <= 0:
            return

        unique_seeds = list(dict.fromkeys(seed_canonicals))
        try:
            seeds = await self._graph.get_concepts(user_id, unique_seeds)
            if not seeds:
                return

            existing_neighbors = await self._collect_existing_neighbors(user_id, seeds)

            seed_mentions = [self._concept_to_mention(c) for c in seeds]
            neighbor_mentions = [self._concept_to_mention(c) for c in existing_neighbors]

            proposal = await self._expert.propose(
                seed_concepts=seed_mentions,
                existing_neighbors=neighbor_mentions,
                max_per_seed=self._max_per_seed,
            )
            if not proposal.relations and not proposal.new_concepts:
                return

            if proposal.new_concepts:
                await self._graph.upsert_concepts(user_id, proposal.new_concepts, initial_score=0)
            if proposal.relations:
                await self._graph.add_relations(user_id, proposal.relations)
        except Exception:  # noqa: BLE001 — background task never raises
            logger.exception("graph expansion failed", extra={"user_id": str(user_id)})

    async def _collect_existing_neighbors(
        self,
        user_id: UUID,
        seeds: Sequence[Concept],
    ) -> list[Concept]:
        seen: set[str] = {s.canonical_name for s in seeds}
        out: list[Concept] = []
        for seed in seeds:
            neighborhood = await self._graph.get_neighborhood(
                user_id, seed.canonical_name, max_hops=1
            )
            if neighborhood is None:
                continue
            for node in neighborhood.nodes:
                if node.canonical_name in seen:
                    continue
                seen.add(node.canonical_name)
                out.append(node)
        return out

    @staticmethod
    def _concept_to_mention(concept: Concept) -> ConceptMention:
        return ConceptMention(
            canonical_name=concept.canonical_name,
            display_name=concept.display_name,
            description=concept.description,
            domain=concept.domain,
        )
