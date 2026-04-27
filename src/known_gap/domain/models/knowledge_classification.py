from dataclasses import dataclass

from src.known_gap.domain.models.concept import Concept, ConceptMention


@dataclass(frozen=True)
class KnowledgeClassification:
    """Partitions extracted concept mentions against a user's graph.

    `known` carries the persisted Concept objects (with known_score) for
    mentions whose canonical name exists in the user's graph AND whose
    score is above the configured threshold. `unknown` is everything else
    — both genuinely new concepts and concepts in the graph but still
    below threshold (i.e. seen but not "owned" yet).
    """

    known: list[Concept]
    unknown: list[ConceptMention]
