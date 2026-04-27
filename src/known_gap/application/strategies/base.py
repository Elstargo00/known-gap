from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.tools import BaseTool

from src.known_gap.application.services.tool_enabled_answerer import ToolEnabledAnswerer
from src.known_gap.domain.models.concept import Concept, ConceptMention
from src.known_gap.domain.models.retrieved_chunk import RetrievedChunk


@dataclass(frozen=True)
class StrategyContext:
    query: str
    retrieved: Sequence[RetrievedChunk]
    known: Sequence[Concept]
    unknown: Sequence[ConceptMention]
    tools: Sequence[BaseTool]


class AskStrategy(ABC):
    """Base class for /ask response strategies. Each subclass owns its
    own system prompt; all strategies share the same tool-enabled
    answerer and may pass tools through to it."""

    def __init__(self, answerer: ToolEnabledAnswerer) -> None:
        self._answerer = answerer

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


def format_known_concepts(known: Sequence[Concept]) -> str:
    if not known:
        return "(none)"
    return ", ".join(f'"{c.display_name}" (score={c.known_score})' for c in known)


def format_unknown_concepts(unknown: Sequence[ConceptMention]) -> str:
    if not unknown:
        return "(none)"
    return ", ".join(f'"{c.display_name}"' for c in unknown)
