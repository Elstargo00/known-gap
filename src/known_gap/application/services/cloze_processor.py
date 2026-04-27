import re
from collections.abc import Sequence

from src.known_gap.domain.models.concept import Concept


class ClozeProcessor:
    """Rewrites a generated answer so that mentions of concepts the user
    already "owns" (known_score > threshold) become inline cloze spans
    the frontend can render as fill-in-the-blanks.

    The transformation is concept-level (not random masking): every
    surface form of a known concept becomes a single
    `<cloze concept="canonical_name">display_form</cloze>` tag. This is
    the only learning-mode mutation — the rest of the answer text is
    untouched, so the user reads naturally everywhere except over the
    points they should be testing themselves on.
    """

    CLOZE_TAG = "cloze"

    def __init__(self, threshold: int) -> None:
        self._threshold = threshold

    def process(self, answer: str, candidates: Sequence[Concept]) -> tuple[str, list[str]]:
        """Apply cloze tagging to `answer` for any candidate whose score
        is strictly above the threshold. Returns (rewritten_answer,
        masked_canonical_names).

        Overlap policy: when two concepts could match the same span, the
        longer display_name wins (so "base case" is preferred over "case").
        Each surface match is wrapped at most once.
        """
        if not answer or not candidates:
            return answer, []

        eligible = [c for c in candidates if c.known_score > self._threshold]
        if not eligible:
            return answer, []

        # Find all candidate matches across all concepts. Longer display
        # names sort earlier so they win on overlap.
        eligible.sort(key=lambda c: len(c.display_name), reverse=True)

        spans: list[tuple[int, int, Concept, str]] = []  # (start, end, concept, surface)
        for concept in eligible:
            if not concept.display_name.strip():
                continue
            pattern = re.compile(
                rf"\b{re.escape(concept.display_name)}\b",
                flags=re.IGNORECASE,
            )
            for match in pattern.finditer(answer):
                spans.append((match.start(), match.end(), concept, match.group(0)))

        if not spans:
            return answer, []

        # Resolve overlaps: sort by start, then by descending length.
        spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
        kept: list[tuple[int, int, Concept, str]] = []
        last_end = -1
        for span in spans:
            start, end, _, _ = span
            if start < last_end:
                continue
            kept.append(span)
            last_end = end

        # Stitch the rewritten answer.
        out: list[str] = []
        cursor = 0
        masked: list[str] = []
        for start, end, concept, surface in kept:
            out.append(answer[cursor:start])
            out.append(
                f'<{self.CLOZE_TAG} concept="{concept.canonical_name}">{surface}</{self.CLOZE_TAG}>'
            )
            masked.append(concept.canonical_name)
            cursor = end
        out.append(answer[cursor:])
        return "".join(out), masked

    @staticmethod
    def strip_tags(text: str) -> str:
        """Remove cloze tags, leaving the original surface text — used
        before re-running concept extraction on the answer."""
        return re.sub(r"<cloze[^>]*>(.*?)</cloze>", r"\1", text, flags=re.DOTALL)
