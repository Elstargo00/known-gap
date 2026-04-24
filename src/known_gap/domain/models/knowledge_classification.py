from dataclasses import dataclass

from src.known_gap.domain.models.concept import ConceptMention


@dataclass(frozen=True)
class KnowledgeClassification:
    known: list[ConceptMention]
    unknown: list[ConceptMention]
