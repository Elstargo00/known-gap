from dataclasses import dataclass


@dataclass(frozen=True)
class ConceptRelation:
    """A typed, directed edge between two concepts in a user's graph.

    `relation_type` is an LLM-defined free-form label such as
    "is_a", "uses", "depends_on", "generalizes", "part_of", etc. It is
    normalized to a snake_case identifier suitable for a Cypher
    relationship type.
    """

    from_canonical: str
    to_canonical: str
    relation_type: str
    rationale: str = ""
