from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ConceptMention:
    """A concept as freshly extracted from text — no historical metadata."""

    canonical_name: str
    display_name: str
    description: str
    domain: str | None


@dataclass(frozen=True)
class Concept:
    """A concept as persisted in a user's knowledge graph.

    `known_score` lives in [0, 100]:
      - 0   → freshly seeded (extracted from a doc / answer, never proven)
      - >0  → bumped each time the user "engages" with the concept in an answer
      - >threshold (default 50) → considered "known" by classification
    """

    canonical_name: str
    display_name: str
    description: str
    domain: str | None
    known_score: int
    first_seen: datetime
    last_seen: datetime
