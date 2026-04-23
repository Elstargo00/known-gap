from collections.abc import Sequence

from src.known_gap.domain.models.retrieved_chunk import RetrievedChunk


class PromptBuilder:
    SYSTEM_PROMPT = (
        "You are a helpful assistant answering questions using the retrieved "
        "context below. Cite sources by their [index] number when you rely on "
        "them. If the context does not contain the answer, say so plainly "
        "rather than guessing."
    )

    def build(
        self,
        query: str,
        retrieved: Sequence[RetrievedChunk],
    ) -> tuple[str, str]:
        if not retrieved:
            user = (
                f"Question: {query}\n\n"
                "No relevant context was retrieved. Answer the question only "
                "if it is general knowledge; otherwise say you do not know."
            )
            return self.SYSTEM_PROMPT, user

        context = "\n\n".join(
            f"[{i + 1}] (source: {chunk.document_filename})\n{chunk.content}"
            for i, chunk in enumerate(retrieved)
        )
        user = f"Context:\n{context}\n\nQuestion: {query}"
        return self.SYSTEM_PROMPT, user
