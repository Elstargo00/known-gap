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
    """A concept as persisted in a user's knowledge graph."""

    canonical_name: str
    display_name: str
    description: str
    domain: str | None
    confidence: float
    times_encountered: int
    first_seen: datetime
    last_seen: datetime
