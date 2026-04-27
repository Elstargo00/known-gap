from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from falkordb.asyncio import FalkorDB

from src.known_gap.domain.models.concept import Concept, ConceptMention
from src.known_gap.domain.models.concept_neighborhood import ConceptNeighborhood
from src.known_gap.domain.models.concept_relation import ConceptRelation
from src.known_gap.domain.repositories.user_graph_repository import UserGraphRepository


class FalkorDBUserGraphRepository(UserGraphRepository):
    """One graph per user, named `user_<uuid.hex>`. Single node label
    `Concept`; single edge label `RELATES_TO` carrying a `relation_type`
    property (LLM-defined, free-form snake_case)."""

    SCORE_MIN = 0
    SCORE_MAX = 100

    def __init__(self, client: FalkorDB) -> None:
        self._client = client

    def _graph_name(self, user_id: UUID) -> str:
        return f"user_{user_id.hex}"

    def _select(self, user_id: UUID) -> Any:
        return self._client.select_graph(self._graph_name(user_id))

    @staticmethod
    def _ts_to_dt(value: Any) -> datetime:
        """FalkorDB returns timestamp() as ms since epoch (int).
        Be lenient — fall back to "now" if anything looks off."""
        try:
            return datetime.fromtimestamp(int(value) / 1000.0, tz=UTC)
        except (TypeError, ValueError):
            return datetime.now(UTC)

    @staticmethod
    def _row_to_concept(props: Mapping[str, Any]) -> Concept:
        return Concept(
            canonical_name=str(props["canonical_name"]),
            display_name=str(props.get("display_name") or props["canonical_name"]),
            description=str(props.get("description") or ""),
            domain=props.get("domain") if props.get("domain") else None,
            known_score=int(props.get("known_score") or 0),
            first_seen=FalkorDBUserGraphRepository._ts_to_dt(props.get("first_seen")),
            last_seen=FalkorDBUserGraphRepository._ts_to_dt(props.get("last_seen")),
        )

    async def upsert_concepts(
        self,
        user_id: UUID,
        mentions: Sequence[ConceptMention],
        initial_score: int = 0,
    ) -> list[str]:
        if not mentions:
            return []

        graph = self._select(user_id)
        names = [m.canonical_name for m in mentions]

        existing_result = await graph.query(
            """
            MATCH (c:Concept) WHERE c.canonical_name IN $names
            RETURN c.canonical_name
            """,
            {"names": names},
        )
        existing: set[str] = {str(row[0]) for row in (existing_result.result_set or [])}

        new_mentions = [m for m in mentions if m.canonical_name not in existing]
        match_payload = [
            {
                "canonical_name": m.canonical_name,
                "display_name": m.display_name,
                "description": m.description,
                "domain": m.domain,
            }
            for m in mentions
            if m.canonical_name in existing
        ]
        create_payload = [
            {
                "canonical_name": m.canonical_name,
                "display_name": m.display_name,
                "description": m.description,
                "domain": m.domain,
            }
            for m in new_mentions
        ]

        if create_payload:
            await graph.query(
                """
                UNWIND $mentions AS m
                CREATE (c:Concept {
                    canonical_name: m.canonical_name,
                    display_name:   m.display_name,
                    description:    m.description,
                    domain:         m.domain,
                    known_score:    $initial,
                    first_seen:     timestamp(),
                    last_seen:      timestamp()
                })
                """,
                {
                    "mentions": create_payload,
                    "initial": max(self.SCORE_MIN, min(self.SCORE_MAX, initial_score)),
                },
            )
        if match_payload:
            await graph.query(
                """
                UNWIND $mentions AS m
                MATCH (c:Concept {canonical_name: m.canonical_name})
                SET c.last_seen    = timestamp(),
                    c.display_name = coalesce(c.display_name, m.display_name),
                    c.description  = coalesce(c.description, m.description),
                    c.domain       = coalesce(c.domain, m.domain)
                """,
                {"mentions": match_payload},
            )
        return [m.canonical_name for m in new_mentions]

    async def get_concepts(
        self,
        user_id: UUID,
        canonical_names: Sequence[str],
    ) -> list[Concept]:
        names = list(canonical_names)
        if not names:
            return []
        graph = self._select(user_id)
        result = await graph.query(
            """
            MATCH (c:Concept) WHERE c.canonical_name IN $names
            RETURN c
            """,
            {"names": names},
        )
        concepts: list[Concept] = []
        for row in result.result_set or []:
            node = row[0]
            props = getattr(node, "properties", node)
            concepts.append(self._row_to_concept(props))
        return concepts

    async def adjust_scores(
        self,
        user_id: UUID,
        deltas: Mapping[str, int],
    ) -> None:
        if not deltas:
            return
        graph = self._select(user_id)
        payload = [{"name": name, "delta": int(delta)} for name, delta in deltas.items()]
        await graph.query(
            """
            UNWIND $rows AS row
            MATCH (c:Concept {canonical_name: row.name})
            SET c.known_score = CASE
                WHEN c.known_score + row.delta < $min THEN $min
                WHEN c.known_score + row.delta > $max THEN $max
                ELSE c.known_score + row.delta
            END,
            c.last_seen = timestamp()
            """,
            {"rows": payload, "min": self.SCORE_MIN, "max": self.SCORE_MAX},
        )

    async def add_relations(
        self,
        user_id: UUID,
        relations: Sequence[ConceptRelation],
    ) -> None:
        if not relations:
            return
        graph = self._select(user_id)
        payload = [
            {
                "from": r.from_canonical,
                "to": r.to_canonical,
                "type": r.relation_type,
                "rationale": r.rationale,
            }
            for r in relations
            if r.from_canonical != r.to_canonical
        ]
        if not payload:
            return
        await graph.query(
            """
            UNWIND $rels AS rel
            MATCH (a:Concept {canonical_name: rel.from})
            MATCH (b:Concept {canonical_name: rel.to})
            MERGE (a)-[r:RELATES_TO {relation_type: rel.type}]->(b)
            ON CREATE SET r.rationale = rel.rationale, r.created_at = timestamp()
            """,
            {"rels": payload},
        )

    async def get_neighborhood(
        self,
        user_id: UUID,
        canonical_name: str,
        max_hops: int,
    ) -> ConceptNeighborhood | None:
        graph = self._select(user_id)
        hops = max(1, int(max_hops))

        center_result = await graph.query(
            "MATCH (c:Concept {canonical_name: $name}) RETURN c LIMIT 1",
            {"name": canonical_name},
        )
        rows = center_result.result_set or []
        if not rows:
            return None
        center_props = getattr(rows[0][0], "properties", rows[0][0])
        center = self._row_to_concept(center_props)

        # Collect distinct neighbor nodes within max_hops.
        nodes_result = await graph.query(
            f"""
            MATCH (c:Concept {{canonical_name: $name}})
            MATCH (c)-[:RELATES_TO*1..{hops}]-(n:Concept)
            WITH DISTINCT n
            RETURN n
            """,
            {"name": canonical_name},
        )
        nodes: list[Concept] = []
        seen: set[str] = {center.canonical_name}
        for row in nodes_result.result_set or []:
            node_props = getattr(row[0], "properties", row[0])
            concept = self._row_to_concept(node_props)
            if concept.canonical_name in seen:
                continue
            seen.add(concept.canonical_name)
            nodes.append(concept)

        # Collect edges among center + neighbors (single-hop edges within scope).
        edges: list[ConceptRelation] = []
        if seen:
            edges_result = await graph.query(
                """
                MATCH (a:Concept)-[r:RELATES_TO]->(b:Concept)
                WHERE a.canonical_name IN $names AND b.canonical_name IN $names
                RETURN a.canonical_name, b.canonical_name, r.relation_type, r.rationale
                """,
                {"names": list(seen)},
            )
            for row in edges_result.result_set or []:
                edges.append(
                    ConceptRelation(
                        from_canonical=str(row[0]),
                        to_canonical=str(row[1]),
                        relation_type=str(row[2] or "related_to"),
                        rationale=str(row[3] or ""),
                    )
                )

        return ConceptNeighborhood(center=center, nodes=nodes, edges=edges)

    async def find_path(
        self,
        user_id: UUID,
        from_canonical: str,
        to_canonical: str,
        max_hops: int,
    ) -> list[Concept] | None:
        graph = self._select(user_id)
        hops = max(1, int(max_hops))
        result = await graph.query(
            f"""
            MATCH (a:Concept {{canonical_name: $from_name}}),
                  (b:Concept {{canonical_name: $to_name}})
            MATCH p = shortestPath((a)-[:RELATES_TO*1..{hops}]-(b))
            RETURN nodes(p) AS nodes
            LIMIT 1
            """,
            {"from_name": from_canonical, "to_name": to_canonical},
        )
        rows = result.result_set or []
        if not rows:
            return None
        nodes_in_path = rows[0][0] or []
        path: list[Concept] = []
        for node in nodes_in_path:
            props = getattr(node, "properties", node)
            path.append(self._row_to_concept(props))
        return path or None
