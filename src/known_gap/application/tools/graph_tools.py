"""LangChain tool definitions wrapping the per-user knowledge graph.

These tools are *portable*: anything that speaks LangChain's
`BaseTool` protocol (LangGraph, MCP-via-LC adapters, the LangChain Hub)
can pick them up. Each tool takes a single bound `(graph, user_id)`
context — the calling LLM never sees user IDs in its tool arguments.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from uuid import UUID

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel

from src.known_gap.domain.repositories.user_graph_repository import UserGraphRepository

DEFAULT_MAX_HOPS = 2


def _concept_to_dict(concept: object) -> dict[str, object]:
    return {
        "canonical_name": getattr(concept, "canonical_name", ""),
        "display_name": getattr(concept, "display_name", ""),
        "description": getattr(concept, "description", ""),
        "domain": getattr(concept, "domain", None),
        "known_score": getattr(concept, "known_score", 0),
    }


def _relation_to_dict(relation: object) -> dict[str, object]:
    return {
        "from": getattr(relation, "from_canonical", ""),
        "to": getattr(relation, "to_canonical", ""),
        "relation_type": getattr(relation, "relation_type", ""),
        "rationale": getattr(relation, "rationale", ""),
    }


class _LookupArgs(BaseModel):
    canonical_names: list[str]


class _NeighborhoodArgs(BaseModel):
    canonical_name: str
    max_hops: int = DEFAULT_MAX_HOPS


class _PathArgs(BaseModel):
    from_canonical: str
    to_canonical: str
    max_hops: int = DEFAULT_MAX_HOPS * 2


def build_graph_tools(
    graph: UserGraphRepository,
    user_id: UUID,
    max_hops: int = DEFAULT_MAX_HOPS,
) -> list[BaseTool]:
    """Build a fresh, request-scoped tool list bound to a single user.

    A new list is constructed per /ask call so that closures capture the
    correct `user_id`. The tool surface is intentionally small and
    read-only — graph writes go through the deterministic post-processing
    path, never through model-driven tool calls (which keeps the system
    safe from prompt-injection-driven graph corruption).
    """

    @tool("lookup_user_known_concepts", args_schema=_LookupArgs)
    async def lookup_user_known_concepts(canonical_names: list[str]) -> str:
        """Return the user's known_score for each concept.

        A score of 0 (or absent from the result) means the user has
        never been exposed to that concept. Scores in (0, 50] mean
        "seen but not owned"; scores > 50 mean "the user knows this".
        Use this to decide whether to re-explain a concept or skip it.

        Args:
            canonical_names: lowercased noun phrases, e.g. ["recursion", "base case"].
        """
        concepts = await graph.get_concepts(user_id, canonical_names)
        by_name = {c.canonical_name: c for c in concepts}
        results = [
            _concept_to_dict(by_name[name])
            if name in by_name
            else {"canonical_name": name, "known_score": 0, "in_graph": False}
            for name in canonical_names
        ]
        return json.dumps({"concepts": results})

    @tool("get_concept_neighborhood", args_schema=_NeighborhoodArgs)
    async def get_concept_neighborhood(
        canonical_name: str,
        max_hops: int = DEFAULT_MAX_HOPS,
    ) -> str:
        """Return the concept and its multi-hop neighbors plus typed edges.

        Use this to ground an answer in what the user has *connected
        knowledge* about — if "recursion" connects to "stack overflow"
        in their graph, the answer can lean on that connection.

        Args:
            canonical_name: lowercased seed concept.
            max_hops: traversal depth (1-4 sensible).
        """
        hops = max(1, min(int(max_hops), 4))
        neighborhood = await graph.get_neighborhood(user_id, canonical_name, hops)
        if neighborhood is None:
            return json.dumps({"in_graph": False, "canonical_name": canonical_name})
        return json.dumps(
            {
                "in_graph": True,
                "center": _concept_to_dict(neighborhood.center),
                "nodes": [_concept_to_dict(n) for n in neighborhood.nodes],
                "edges": [_relation_to_dict(e) for e in neighborhood.edges],
                "max_hops": hops,
            }
        )

    @tool("find_path_between_concepts", args_schema=_PathArgs)
    async def find_path_between_concepts(
        from_canonical: str,
        to_canonical: str,
        max_hops: int = DEFAULT_MAX_HOPS * 2,
    ) -> str:
        """Find the shortest path between two concepts in the user's graph.

        Useful for multi-hop reasoning: "how is X related to Y in what
        the user already knows?". Returns the ordered chain of concepts
        and a `path_length` of -1 when there is no path within max_hops.

        Args:
            from_canonical: source concept (lowercased).
            to_canonical: destination concept (lowercased).
            max_hops: max traversal depth.
        """
        hops = max(1, min(int(max_hops), 6))
        path = await graph.find_path(user_id, from_canonical, to_canonical, hops)
        if not path:
            return json.dumps(
                {
                    "from": from_canonical,
                    "to": to_canonical,
                    "path_length": -1,
                    "path": [],
                }
            )
        return json.dumps(
            {
                "from": from_canonical,
                "to": to_canonical,
                "path_length": len(path) - 1,
                "path": [_concept_to_dict(c) for c in path],
            }
        )

    return [
        lookup_user_known_concepts,
        get_concept_neighborhood,
        find_path_between_concepts,
    ]


def langchain_tools_to_anthropic_schema(tools: Sequence[BaseTool]) -> list[dict[str, object]]:
    """Convert LangChain BaseTool definitions to Anthropic's tool schema.

    Anthropic's Messages API expects:
        {"name": str, "description": str, "input_schema": JSONSchema}.

    LangChain exposes `args_schema` (a Pydantic model) plus `.name` and
    `.description`, which we lift directly.
    """
    out: list[dict[str, object]] = []
    for t in tools:
        schema_model = t.args_schema
        if isinstance(schema_model, type) and issubclass(schema_model, BaseModel):
            input_schema = schema_model.model_json_schema()
        else:
            input_schema = {"type": "object", "properties": {}}
        out.append(
            {
                "name": t.name,
                "description": t.description or "",
                "input_schema": input_schema,
            }
        )
    return out
