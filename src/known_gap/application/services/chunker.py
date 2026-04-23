from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    content: str
    index: int


class Chunker:
    """Recursive character text splitter.

    Walks a ladder of separators and splits on the largest one that keeps
    pieces within the size budget. Falls back to character-level splits
    only for continuous text with no separators (e.g. a single very long
    word). Overlap is best-effort: whole tail pieces are carried into the
    next chunk up to the overlap budget — boundary preservation beats
    exact overlap accounting.
    """

    DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        separators: tuple[str, ...] | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separators = separators or self.DEFAULT_SEPARATORS

    def chunk(self, text: str) -> list[TextChunk]:
        if not text.strip():
            return []
        pieces = self._split(text, self._separators)
        merged = self._merge(pieces)
        return [TextChunk(content=content, index=idx) for idx, content in enumerate(merged)]

    def _split(self, text: str, separators: tuple[str, ...]) -> list[str]:
        separator, remaining = self._pick_separator(text, separators)

        if separator:
            parts = text.split(separator)
            pieces = [p + separator for p in parts[:-1]] + [parts[-1]]
        else:
            pieces = list(text)

        result: list[str] = []
        for piece in pieces:
            if not piece:
                continue
            if len(piece) <= self._chunk_size:
                result.append(piece)
            elif remaining:
                result.extend(self._split(piece, remaining))
            else:
                result.extend(
                    piece[i : i + self._chunk_size] for i in range(0, len(piece), self._chunk_size)
                )
        return result

    def _pick_separator(
        self, text: str, separators: tuple[str, ...]
    ) -> tuple[str, tuple[str, ...]]:
        for i, sep in enumerate(separators):
            if sep == "" or sep in text:
                return sep, separators[i + 1 :]
        return "", ()

    def _merge(self, pieces: list[str]) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        def flush() -> None:
            nonlocal current, current_len
            joined = "".join(current).strip()
            if joined:
                chunks.append(joined)
            overlap: list[str] = []
            overlap_len = 0
            for item in reversed(current):
                if overlap_len + len(item) > self._chunk_overlap:
                    break
                overlap.insert(0, item)
                overlap_len += len(item)
            current = overlap
            current_len = overlap_len

        for piece in pieces:
            if current_len + len(piece) > self._chunk_size and current:
                flush()
            current.append(piece)
            current_len += len(piece)

        tail = "".join(current).strip()
        if tail:
            chunks.append(tail)
        return chunks
