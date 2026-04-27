from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import uuid4

from src.known_gap.application.services.graph_expander import GraphExpander
from src.known_gap.domain.models.concept import Concept, ConceptMention
from src.known_gap.domain.models.concept_relation import ConceptRelation
from src.known_gap.domain.services.graph_expert import ExpertProposal, GraphExpert
from tests.unit._fakes import FakeUserGraphRepository


class _StubExpert(GraphExpert):
    def __init__(self, proposal: ExpertProposal | Exception) -> None:
        self._proposal = proposal
        self.calls: list[tuple[Sequence[ConceptMention], Sequence[ConceptMention], int]] = []

    async def propose(
        self,
        seed_concepts: Sequence[ConceptMention],
        existing_neighbors: Sequence[ConceptMention],
        max_per_seed: int,
    ) -> ExpertProposal:
        self.calls.append((list(seed_concepts), list(existing_neighbors), max_per_seed))
        if isinstance(self._proposal, Exception):
            raise self._proposal
        return self._proposal


def _seed_concept(graph: FakeUserGraphRepository, user_id, name: str, score: int = 0) -> None:
    bucket = graph._user_concepts(user_id)  # noqa: SLF001 — test seam
    now = datetime.now(UTC)
    bucket[name] = Concept(
        canonical_name=name,
        display_name=name.title(),
        description="",
        domain=None,
        known_score=score,
        first_seen=now,
        last_seen=now,
    )


class TestGraphExpander:
    async def test_persists_new_concepts_and_relations(self) -> None:
        user_id = uuid4()
        graph = FakeUserGraphRepository()
        _seed_concept(graph, user_id, "recursion")

        proposal = ExpertProposal(
            new_concepts=[
                ConceptMention(
                    canonical_name="stack frame",
                    display_name="Stack frame",
                    description="",
                    domain=None,
                ),
            ],
            relations=[
                ConceptRelation(
                    from_canonical="recursion",
                    to_canonical="stack frame",
                    relation_type="uses",
                ),
            ],
        )
        expert = _StubExpert(proposal)
        expander = GraphExpander(graph=graph, expert=expert, max_per_seed=3)

        await expander.expand(user_id, ["recursion"])

        # The new concept was upserted with score=0 first, so the
        # relation can land.
        assert "stack frame" in graph._user_concepts(user_id)
        assert graph._user_concepts(user_id)["stack frame"].known_score == 0
        assert any(
            r.from_canonical == "recursion" and r.to_canonical == "stack frame"
            for r in graph._user_relations(user_id)
        )

    async def test_swallows_expert_failure(self) -> None:
        user_id = uuid4()
        graph = FakeUserGraphRepository()
        _seed_concept(graph, user_id, "recursion")
        expander = GraphExpander(
            graph=graph,
            expert=_StubExpert(RuntimeError("boom")),
            max_per_seed=3,
        )

        # Must not raise. Background task semantics.
        await expander.expand(user_id, ["recursion"])
        assert graph._user_relations(user_id) == []

    async def test_noop_for_unknown_seed(self) -> None:
        user_id = uuid4()
        graph = FakeUserGraphRepository()
        expert = _StubExpert(ExpertProposal(new_concepts=[], relations=[]))
        expander = GraphExpander(graph=graph, expert=expert, max_per_seed=3)

        await expander.expand(user_id, ["unknown"])

        # The expert was never invoked because no seed exists in the graph.
        assert expert.calls == []

    async def test_includes_existing_neighbors_in_expert_prompt(self) -> None:
        user_id = uuid4()
        graph = FakeUserGraphRepository()
        _seed_concept(graph, user_id, "recursion")
        _seed_concept(graph, user_id, "loop")
        await graph.add_relations(
            user_id,
            [
                ConceptRelation(
                    from_canonical="recursion",
                    to_canonical="loop",
                    relation_type="contrasts_with",
                ),
            ],
        )
        expert = _StubExpert(ExpertProposal(new_concepts=[], relations=[]))
        expander = GraphExpander(graph=graph, expert=expert, max_per_seed=3)

        await expander.expand(user_id, ["recursion"])

        assert len(expert.calls) == 1
        seeds, neighbors, max_per_seed = expert.calls[0]
        assert {s.canonical_name for s in seeds} == {"recursion"}
        assert {n.canonical_name for n in neighbors} == {"loop"}
        assert max_per_seed == 3
