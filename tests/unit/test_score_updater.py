from datetime import UTC, datetime
from uuid import uuid4

from src.known_gap.application.services.score_updater import ScoreUpdater
from src.known_gap.domain.models.concept import Concept, ConceptMention
from tests.unit._fakes import FakeUserGraphRepository


def _mention(name: str) -> ConceptMention:
    return ConceptMention(
        canonical_name=name.lower(),
        display_name=name,
        description="",
        domain=None,
    )


def _concept(name: str, score: int) -> Concept:
    now = datetime.now(UTC)
    return Concept(
        canonical_name=name.lower(),
        display_name=name,
        description="",
        domain=None,
        known_score=score,
        first_seen=now,
        last_seen=now,
    )


def _seed(graph: FakeUserGraphRepository, user_id, scores: dict[str, int]) -> None:
    bucket = graph._user_concepts(user_id)  # noqa: SLF001 — test seam
    now = datetime.now(UTC)
    for name, score in scores.items():
        bucket[name] = Concept(
            canonical_name=name,
            display_name=name.title(),
            description="",
            domain=None,
            known_score=score,
            first_seen=now,
            last_seen=now,
        )


class TestScoreUpdater:
    async def test_re_asked_known_concept_is_decremented(self) -> None:
        user_id = uuid4()
        graph = FakeUserGraphRepository()
        _seed(graph, user_id, {"recursion": 70})
        updater = ScoreUpdater(graph, threshold=50, increment=10, decrement=10)

        deltas = await updater.apply(
            user_id=user_id,
            question_concepts=[_mention("Recursion")],
            answer_concepts=[_mention("Recursion")],
            known_in_question=[_concept("Recursion", 70)],
        )

        assert deltas == {"recursion": -10}
        assert graph._user_concepts(user_id)["recursion"].known_score == 60

    async def test_introduced_in_answer_only_is_incremented(self) -> None:
        user_id = uuid4()
        graph = FakeUserGraphRepository()
        _seed(graph, user_id, {"recursion": 30, "stack frame": 0})
        updater = ScoreUpdater(graph, threshold=50, increment=10, decrement=10)

        deltas = await updater.apply(
            user_id=user_id,
            question_concepts=[_mention("Recursion")],
            answer_concepts=[_mention("Recursion"), _mention("Stack frame")],
            known_in_question=[],
        )

        # "recursion" is in both Q and A → no change. "stack frame" only in A → +10.
        assert deltas == {"stack frame": 10}
        assert graph._user_concepts(user_id)["stack frame"].known_score == 10
        assert graph._user_concepts(user_id)["recursion"].known_score == 30

    async def test_score_clamps_to_bounds(self) -> None:
        user_id = uuid4()
        graph = FakeUserGraphRepository()
        _seed(graph, user_id, {"recursion": 95, "trampolining": 5})
        updater = ScoreUpdater(graph, threshold=50, increment=10, decrement=10)

        await updater.apply(
            user_id=user_id,
            question_concepts=[_mention("Trampolining")],  # not in known_in_question
            answer_concepts=[_mention("Recursion")],
            known_in_question=[],
        )

        # recursion only in A → +10 capped at 100.
        assert graph._user_concepts(user_id)["recursion"].known_score == 100

    async def test_decrement_can_go_below_zero_then_clamps(self) -> None:
        user_id = uuid4()
        graph = FakeUserGraphRepository()
        # An (unrealistic) edge case: known_score=55 and decrement=100.
        _seed(graph, user_id, {"recursion": 55})
        updater = ScoreUpdater(graph, threshold=50, increment=10, decrement=100)

        await updater.apply(
            user_id=user_id,
            question_concepts=[_mention("Recursion")],
            answer_concepts=[_mention("Recursion")],
            known_in_question=[_concept("Recursion", 55)],
        )

        assert graph._user_concepts(user_id)["recursion"].known_score == 0

    async def test_below_threshold_known_concept_is_not_decremented(self) -> None:
        user_id = uuid4()
        graph = FakeUserGraphRepository()
        _seed(graph, user_id, {"recursion": 40})
        updater = ScoreUpdater(graph, threshold=50, increment=10, decrement=10)

        # "Recursion" exists but score=40 → not "known", so it does not
        # count as a known_in_question entry per classifier rules. We
        # exercise the updater contract directly: empty known_in_question.
        deltas = await updater.apply(
            user_id=user_id,
            question_concepts=[_mention("Recursion")],
            answer_concepts=[_mention("Recursion")],
            known_in_question=[],
        )
        assert deltas == {}
        assert graph._user_concepts(user_id)["recursion"].known_score == 40
