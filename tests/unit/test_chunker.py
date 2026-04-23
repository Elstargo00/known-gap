import pytest

from src.known_gap.application.services.chunker import Chunker


class TestChunker:
    def test_empty_text_returns_no_chunks(self) -> None:
        chunker = Chunker(chunk_size=100, chunk_overlap=10)
        assert chunker.chunk("") == []

    def test_whitespace_only_returns_no_chunks(self) -> None:
        chunker = Chunker(chunk_size=100, chunk_overlap=10)
        assert chunker.chunk("   \n\t  ") == []

    def test_short_text_produces_single_chunk(self) -> None:
        chunker = Chunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk("hello world")
        assert len(chunks) == 1
        assert chunks[0].content == "hello world"
        assert chunks[0].index == 0

    def test_long_text_is_split_with_sequential_indexes(self) -> None:
        chunker = Chunker(chunk_size=10, chunk_overlap=2)
        chunks = chunker.chunk("a" * 25)
        assert len(chunks) >= 3
        assert all(len(c.content) <= 10 for c in chunks)
        assert [c.index for c in chunks] == list(range(len(chunks)))

    def test_overlap_preserves_context_between_chunks(self) -> None:
        chunker = Chunker(chunk_size=10, chunk_overlap=4)
        text = "abcdefghijklmnop"
        chunks = chunker.chunk(text)
        # Second chunk starts at step = 6, so its first 4 chars overlap
        # with the last 4 chars of the first chunk.
        assert len(chunks) >= 2
        assert text[6:10] in chunks[0].content
        assert text[6:10] in chunks[1].content

    def test_invalid_overlap_raises(self) -> None:
        with pytest.raises(ValueError):
            Chunker(chunk_size=10, chunk_overlap=10)

    def test_negative_overlap_raises(self) -> None:
        with pytest.raises(ValueError):
            Chunker(chunk_size=10, chunk_overlap=-1)

    def test_non_positive_chunk_size_raises(self) -> None:
        with pytest.raises(ValueError):
            Chunker(chunk_size=0, chunk_overlap=0)
