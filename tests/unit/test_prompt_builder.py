from uuid import uuid4

from src.known_gap.application.services.prompt_builder import PromptBuilder
from src.known_gap.domain.models.retrieved_chunk import RetrievedChunk


def _chunk(content: str, filename: str = "doc.md") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        content=content,
        document_filename=filename,
        similarity_score=0.9,
    )


class TestPromptBuilder:
    def test_builds_context_with_citations(self) -> None:
        chunks = [
            _chunk("Python is a programming language.", "intro.md"),
            _chunk("Python is interpreted.", "intro.md"),
        ]
        system, user = PromptBuilder().build("What is Python?", chunks)

        assert "helpful assistant" in system.lower()
        assert "Python is a programming language." in user
        assert "Python is interpreted." in user
        assert "[1]" in user
        assert "[2]" in user
        assert "intro.md" in user
        assert "What is Python?" in user

    def test_empty_retrieval_produces_no_context_prompt(self) -> None:
        system, user = PromptBuilder().build("What is Python?", [])
        assert "No relevant context" in user
        assert "What is Python?" in user
        assert system  # system prompt still present

    def test_citation_indexes_start_at_one(self) -> None:
        chunks = [_chunk("first"), _chunk("second"), _chunk("third")]
        _, user = PromptBuilder().build("q", chunks)
        assert "[1]" in user
        assert "[2]" in user
        assert "[3]" in user
        assert "[0]" not in user
