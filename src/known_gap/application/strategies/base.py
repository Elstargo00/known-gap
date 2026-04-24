from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from src.known_gap.domain.models.concept import ConceptMention
from src.known_gap.domain.models.retrieved_chunk import RetrievedChunk
from src.known_gap.domain.services.llm_provider import LLMProvider


@dataclass(frozen=True)
class StrategyContext:
    query: str
    retrieved: Sequence[RetrievedChunk]
    known: Sequence[ConceptMention]
    unknown: Sequence[ConceptMention]


class AskStrategy(ABC):
    def __init__(self, llm: LLMProvider, max_tokens: int) -> None:
        self._llm = llm
        self._max_tokens = max_tokens

    @abstractmethod
    async def answer(self, context: StrategyContext) -> str: ...


def format_retrieved_context(retrieved: Sequence[RetrievedChunk]) -> str:
    """Numbered, source-labelled block of retrieved chunks for citations."""
    if not retrieved:
        return ""
    return "\n\n".join(
        f"[{index + 1}] (source: {chunk.document_filename})\n{chunk.content}"
        for index, chunk in enumerate(retrieved)
    )


def format_known_concepts(known: Sequence[ConceptMention]) -> str:
    if not known:
        return "(none)"
    return ", ".join(f'"{c.display_name}"' for c in known)
