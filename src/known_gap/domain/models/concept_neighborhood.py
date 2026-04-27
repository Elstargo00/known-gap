from dataclasses import dataclass, field

from src.known_gap.domain.models.concept import Concept
from src.known_gap.domain.models.concept_relation import ConceptRelation


@dataclass(frozen=True)
class ConceptNeighborhood:
    """A concept with its multi-hop neighbors and the connecting edges.

    Returned by `UserGraphRepository.get_neighborhood`. The center is the
    seed concept; `nodes` are concepts reachable within `max_hops`;
    `edges` are the relations traversed (de-duplicated).
    """

    center: Concept
    nodes: list[Concept] = field(default_factory=list)
    edges: list[ConceptRelation] = field(default_factory=list)
