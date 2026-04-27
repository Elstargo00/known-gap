from datetime import UTC, datetime

from src.known_gap.application.services.cloze_processor import ClozeProcessor
from src.known_gap.domain.models.concept import Concept


def _concept(name: str, display: str, score: int) -> Concept:
    now = datetime.now(UTC)
    return Concept(
        canonical_name=name,
        display_name=display,
        description="",
        domain=None,
        known_score=score,
        first_seen=now,
        last_seen=now,
    )


class TestClozeProcessor:
    def test_wraps_known_concepts_above_threshold(self) -> None:
        processor = ClozeProcessor(threshold=50)
        text = "Recursion uses a base case to terminate."
        candidates = [
            _concept("recursion", "Recursion", 80),
            _concept("base case", "base case", 70),
        ]

        out, masked = processor.process(text, candidates)

        assert '<cloze concept="recursion">Recursion</cloze>' in out
        assert '<cloze concept="base case">base case</cloze>' in out
        assert set(masked) == {"recursion", "base case"}

    def test_skips_candidates_below_or_equal_threshold(self) -> None:
        processor = ClozeProcessor(threshold=50)
        text = "Memoisation caches results."
        candidates = [_concept("memoisation", "Memoisation", 50)]

        out, masked = processor.process(text, candidates)

        assert out == text
        assert masked == []

    def test_longer_concept_wins_on_overlap(self) -> None:
        processor = ClozeProcessor(threshold=50)
        text = "The base case stops the recursion."
        candidates = [
            _concept("case", "case", 80),
            _concept("base case", "base case", 80),
        ]

        out, _ = processor.process(text, candidates)

        # "base case" must win — there should be no nested or duplicate cloze.
        assert '<cloze concept="base case">base case</cloze>' in out
        assert out.count("<cloze") == 1

    def test_case_insensitive_match_preserves_original_surface(self) -> None:
        processor = ClozeProcessor(threshold=50)
        text = "RECURSION is powerful."
        candidates = [_concept("recursion", "recursion", 70)]

        out, _ = processor.process(text, candidates)

        assert '<cloze concept="recursion">RECURSION</cloze>' in out

    def test_strip_tags_round_trips(self) -> None:
        text = 'Recursion uses <cloze concept="base-case">a base case</cloze> to terminate.'
        assert ClozeProcessor.strip_tags(text) == "Recursion uses a base case to terminate."

    def test_empty_inputs_return_unchanged(self) -> None:
        processor = ClozeProcessor(threshold=50)
        assert processor.process("", []) == ("", [])
        assert processor.process("hello", []) == ("hello", [])
