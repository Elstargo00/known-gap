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
        assert len(chunks) >= 2
        assert text[6:10] in chunks[0].content
        assert text[6:10] in chunks[1].content

    def test_respects_word_boundaries_when_possible(self) -> None:
        chunker = Chunker(chunk_size=25, chunk_overlap=5)
        text = "the quick brown fox jumps over the lazy dog very fast"
        source_words = set(text.split())
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2
        for chunk in chunks:
            for word in chunk.content.split():
                assert word in source_words, f"partial word {word!r} in {chunk.content!r}"

    def test_prefers_paragraph_then_sentence_boundaries(self) -> None:
        chunker = Chunker(chunk_size=30, chunk_overlap=5)
        text = "Alpha paragraph.\n\nBeta paragraph.\n\nGamma paragraph."
        chunks = chunker.chunk(text)
        # Each chunk content must be reconstructible from whole words only.
        source_words = set(text.replace("\n\n", " ").split())
        for chunk in chunks:
            for word in chunk.content.split():
                assert word in source_words

    def test_falls_back_to_char_split_for_long_unbroken_words(self) -> None:
        chunker = Chunker(chunk_size=5, chunk_overlap=1)
        chunks = chunker.chunk("supercalifragilistic")
        assert len(chunks) > 1
        assert all(len(c.content) <= 5 for c in chunks)

    def test_invalid_overlap_raises(self) -> None:
        with pytest.raises(ValueError):
            Chunker(chunk_size=10, chunk_overlap=10)

    def test_negative_overlap_raises(self) -> None:
        with pytest.raises(ValueError):
            Chunker(chunk_size=10, chunk_overlap=-1)

    def test_non_positive_chunk_size_raises(self) -> None:
        with pytest.raises(ValueError):
            Chunker(chunk_size=0, chunk_overlap=0)
