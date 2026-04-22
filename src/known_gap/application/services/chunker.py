from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    content: str
    index: int


class Chunker:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> list[TextChunk]:
        if not text.strip():
            return []

        step = self._chunk_size - self._chunk_overlap
        chunks: list[TextChunk] = []
        start = 0
        while start < len(text):
            content = text[start : start + self._chunk_size].strip()
            if content:
                chunks.append(TextChunk(content=content, index=len(chunks)))
            start += step
        return chunks
