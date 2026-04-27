"""Shared in-memory fakes for unit tests.

These fakes intentionally cover all UserGraphRepository methods so any
test can stand them up without hand-rolling stubs."""

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from uuid import UUID

from src.known_gap.domain.models.concept import Concept, ConceptMention
from src.known_gap.domain.models.concept_neighborhood import ConceptNeighborhood
from src.known_gap.domain.models.concept_relation import ConceptRelation
from src.known_gap.domain.repositories.user_graph_repository import UserGraphRepository
from src.known_gap.domain.services.llm_provider import LLMProvider


class FakeLLM(LLMProvider):
    def __init__(self, *responses: str) -> None:
        self._responses = list(responses) or [""]
        self.calls: list[tuple[str, str, int]] = []

    async def complete(self, system: str, user: str, max_tokens: int) -> str:
        self.calls.append((system, user, max_tokens))
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


class FakeUserGraphRepository(UserGraphRepository):
    """In-memory user graph: single dict of users → {concepts, relations}."""

    def __init__(self) -> None:
        self.concepts: dict[UUID, dict[str, Concept]] = {}
        self.relations: dict[UUID, list[ConceptRelation]] = {}
        self.upsert_calls: list[tuple[UUID, list[ConceptMention], int]] = []
        self.adjust_calls: list[tuple[UUID, dict[str, int]]] = []
        self.add_relations_calls: list[tuple[UUID, list[ConceptRelation]]] = []

    def _user_concepts(self, user_id: UUID) -> dict[str, Concept]:
        return self.concepts.setdefault(user_id, {})

    def _user_relations(self, user_id: UUID) -> list[ConceptRelation]:
        return self.relations.setdefault(user_id, [])

    async def upsert_concepts(
        self,
        user_id: UUID,
        mentions: Sequence[ConceptMention],
        initial_score: int = 0,
    ) -> list[str]:
        self.upsert_calls.append((user_id, list(mentions), initial_score))
        bucket = self._user_concepts(user_id)
        now = datetime.now(UTC)
        newly_created: list[str] = []
        for m in mentions:
            if m.canonical_name in bucket:
                existing = bucket[m.canonical_name]
                bucket[m.canonical_name] = Concept(
                    canonical_name=existing.canonical_name,
                    display_name=existing.display_name or m.display_name,
                    description=existing.description or m.description,
                    domain=existing.domain or m.domain,
                    known_score=existing.known_score,
                    first_seen=existing.first_seen,
                    last_seen=now,
                )
            else:
                bucket[m.canonical_name] = Concept(
                    canonical_name=m.canonical_name,
                    display_name=m.display_name,
                    description=m.description,
                    domain=m.domain,
                    known_score=max(0, min(100, initial_score)),
                    first_seen=now,
                    last_seen=now,
                )
                newly_created.append(m.canonical_name)
        return newly_created

    async def get_concepts(
        self,
        user_id: UUID,
        canonical_names: Sequence[str],
    ) -> list[Concept]:
        bucket = self._user_concepts(user_id)
        return [bucket[name] for name in canonical_names if name in bucket]

    async def adjust_scores(
        self,
        user_id: UUID,
        deltas: Mapping[str, int],
    ) -> None:
        self.adjust_calls.append((user_id, dict(deltas)))
        bucket = self._user_concepts(user_id)
        for name, delta in deltas.items():
            if name not in bucket:
                continue
            existing = bucket[name]
            new_score = max(0, min(100, existing.known_score + delta))
            bucket[name] = Concept(
                canonical_name=existing.canonical_name,
                display_name=existing.display_name,
                description=existing.description,
                domain=existing.domain,
                known_score=new_score,
                first_seen=existing.first_seen,
                last_seen=datetime.now(UTC),
            )

    async def add_relations(
        self,
        user_id: UUID,
        relations: Sequence[ConceptRelation],
    ) -> None:
        self.add_relations_calls.append((user_id, list(relations)))
        existing = self._user_relations(user_id)
        bucket = self._user_concepts(user_id)
        seen = {(r.from_canonical, r.to_canonical, r.relation_type) for r in existing}
        for rel in relations:
            if (
                rel.from_canonical not in bucket
                or rel.to_canonical not in bucket
                or rel.from_canonical == rel.to_canonical
            ):
                continue
            key = (rel.from_canonical, rel.to_canonical, rel.relation_type)
            if key in seen:
                continue
            seen.add(key)
            existing.append(rel)

    async def get_neighborhood(
        self,
        user_id: UUID,
        canonical_name: str,
        max_hops: int,
    ) -> ConceptNeighborhood | None:
        bucket = self._user_concepts(user_id)
        if canonical_name not in bucket:
            return None
        center = bucket[canonical_name]
        rels = self._user_relations(user_id)

        # BFS up to max_hops
        seen: set[str] = {canonical_name}
        frontier: set[str] = {canonical_name}
        for _ in range(max(1, int(max_hops))):
            nxt: set[str] = set()
            for rel in rels:
                if rel.from_canonical in frontier and rel.to_canonical not in seen:
                    nxt.add(rel.to_canonical)
                if rel.to_canonical in frontier and rel.from_canonical not in seen:
                    nxt.add(rel.from_canonical)
            seen |= nxt
            frontier = nxt
            if not frontier:
                break

        nodes = [bucket[n] for n in seen if n != canonical_name and n in bucket]
        edges = [r for r in rels if r.from_canonical in seen and r.to_canonical in seen]
        return ConceptNeighborhood(center=center, nodes=nodes, edges=edges)

    async def find_path(
        self,
        user_id: UUID,
        from_canonical: str,
        to_canonical: str,
        max_hops: int,
    ) -> list[Concept] | None:
        bucket = self._user_concepts(user_id)
        if from_canonical not in bucket or to_canonical not in bucket:
            return None
        rels = self._user_relations(user_id)
        # BFS predecessor map for shortest path (undirected for traversal).
        prev: dict[str, str] = {}
        frontier = [from_canonical]
        visited = {from_canonical}
        for _ in range(max(1, int(max_hops))):
            new_frontier: list[str] = []
            for node in frontier:
                neighbors = {r.to_canonical for r in rels if r.from_canonical == node}
                neighbors |= {r.from_canonical for r in rels if r.to_canonical == node}
                for n in neighbors - visited:
                    visited.add(n)
                    prev[n] = node
                    new_frontier.append(n)
            frontier = new_frontier
            if to_canonical in visited:
                break
        if to_canonical not in visited:
            return None
        path_names: list[str] = [to_canonical]
        cur = to_canonical
        while cur != from_canonical:
            cur = prev[cur]
            path_names.append(cur)
        path_names.reverse()
        return [bucket[n] for n in path_names]
