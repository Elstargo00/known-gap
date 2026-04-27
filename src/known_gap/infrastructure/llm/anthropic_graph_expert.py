import json
import re
from collections.abc import Sequence
from typing import Any

from src.known_gap.domain.models.concept import ConceptMention
from src.known_gap.domain.models.concept_relation import ConceptRelation
from src.known_gap.domain.services.graph_expert import ExpertProposal, GraphExpert
from src.known_gap.domain.services.llm_provider import LLMProvider

_RELATION_TYPE_PATTERN = re.compile(r"[^a-z0-9_]+")


class AnthropicGraphExpert(GraphExpert):
    """LLM-backed graph expert. Reuses the project's LLMProvider port so
    we get the same fallback semantics (Anthropic primary, Gemini on
    transient errors). The "Anthropic" in the class name reflects the
    *intent* (run with a high-capability Anthropic model like Opus) but
    the implementation is provider-agnostic."""

    SYSTEM_PROMPT = (
        "You are a knowledge-graph expert. Given a set of seed concepts and "
        "their existing graph neighbors, you propose new typed, directed "
        "relations that densify the graph using your world knowledge.\n\n"
        "Return ONLY a JSON object with this exact structure:\n"
        '{"new_concepts": [{"name": "...", "description": "...", "domain": "..."}],\n'
        ' "relations": [{"from": "<canonical_lowercase>", "to": "<canonical_lowercase>", '
        '"relation_type": "<snake_case verb phrase>", "rationale": "<one sentence>"}]}\n\n'
        "Rules:\n"
        "- Propose AT MOST `max_per_seed` outgoing relations per seed concept.\n"
        "- Prefer connecting to existing neighbors when sensible; otherwise "
        "introduce new concepts.\n"
        "- relation_type: short, lower_snake_case verb phrase like 'is_a', "
        "'uses', 'depends_on', 'generalizes', 'part_of', 'contrasts_with'.\n"
        "- Do NOT duplicate existing edges.\n"
        "- canonical names are lowercased noun phrases (2-4 words max).\n"
        "- Return JSON only. No prose. No markdown fences."
    )

    def __init__(self, llm: LLMProvider, max_tokens: int = 2048) -> None:
        self._llm = llm
        self._max_tokens = max_tokens

    async def propose(
        self,
        seed_concepts: Sequence[ConceptMention],
        existing_neighbors: Sequence[ConceptMention],
        max_per_seed: int,
    ) -> ExpertProposal:
        if not seed_concepts or max_per_seed <= 0:
            return ExpertProposal(new_concepts=[], relations=[])

        user = self._build_user_prompt(seed_concepts, existing_neighbors, max_per_seed)
        try:
            raw = await self._llm.complete(
                system=self.SYSTEM_PROMPT,
                user=user,
                max_tokens=self._max_tokens,
            )
        except Exception:
            # Background-task semantics: never let an expert failure
            # cascade. Return an empty proposal so the caller logs and
            # moves on.
            return ExpertProposal(new_concepts=[], relations=[])

        return self._parse(raw, seed_canonicals={c.canonical_name for c in seed_concepts})

    def _build_user_prompt(
        self,
        seeds: Sequence[ConceptMention],
        neighbors: Sequence[ConceptMention],
        max_per_seed: int,
    ) -> str:
        seeds_block = (
            "\n".join(f"- {s.canonical_name} :: {s.display_name} — {s.description}" for s in seeds)
            or "(none)"
        )
        neighbors_block = (
            "\n".join(f"- {n.canonical_name} :: {n.display_name}" for n in neighbors) or "(none)"
        )
        return (
            f"max_per_seed = {max_per_seed}\n\n"
            f"Seed concepts:\n{seeds_block}\n\n"
            f"Existing neighbors (already in the user's graph):\n{neighbors_block}\n\n"
            "Propose new_concepts and relations. JSON only."
        )

    @staticmethod
    def _normalize_relation_type(raw_type: str) -> str:
        cleaned = _RELATION_TYPE_PATTERN.sub("_", raw_type.strip().lower()).strip("_")
        return cleaned or "related_to"

    def _parse(self, raw: str, seed_canonicals: set[str]) -> ExpertProposal:
        text = self._strip_fences(raw)
        try:
            payload: Any = json.loads(text)
        except json.JSONDecodeError:
            return ExpertProposal(new_concepts=[], relations=[])
        if not isinstance(payload, dict):
            return ExpertProposal(new_concepts=[], relations=[])

        new_concepts: list[ConceptMention] = []
        for item in payload.get("new_concepts", []) or []:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            display = name.strip()
            description = item.get("description") or ""
            domain = item.get("domain")
            new_concepts.append(
                ConceptMention(
                    canonical_name=display.lower(),
                    display_name=display,
                    description=description if isinstance(description, str) else "",
                    domain=domain if isinstance(domain, str) and domain.strip() else None,
                )
            )

        relations: list[ConceptRelation] = []
        for item in payload.get("relations", []) or []:
            if not isinstance(item, dict):
                continue
            src = item.get("from")
            dst = item.get("to")
            if not isinstance(src, str) or not isinstance(dst, str):
                continue
            src_norm = src.strip().lower()
            dst_norm = dst.strip().lower()
            if not src_norm or not dst_norm or src_norm == dst_norm:
                continue
            rel_type = self._normalize_relation_type(str(item.get("relation_type") or ""))
            rationale = item.get("rationale") or ""
            relations.append(
                ConceptRelation(
                    from_canonical=src_norm,
                    to_canonical=dst_norm,
                    relation_type=rel_type,
                    rationale=rationale if isinstance(rationale, str) else "",
                )
            )

        # Light sanity gate: drop relations whose endpoints reference
        # neither a seed nor a freshly-proposed new concept on the
        # *source* side — keeps the expert honest (every edge must
        # ground in something we asked about).
        new_canonicals = {c.canonical_name for c in new_concepts}
        valid_sources = seed_canonicals | new_canonicals
        relations = [r for r in relations if r.from_canonical in valid_sources]

        return ExpertProposal(new_concepts=new_concepts, relations=relations)

    @staticmethod
    def _strip_fences(raw: str) -> str:
        text = raw.strip()
        if not text.startswith("```"):
            return text
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
