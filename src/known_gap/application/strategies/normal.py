from src.known_gap.application.strategies.base import (
    AskStrategy,
    StrategyContext,
    format_retrieved_context,
)


class NormalStrategy(AskStrategy):
    SYSTEM_PROMPT = (
        "You are a helpful assistant answering questions using the retrieved "
        "context below. Cite sources by their [index] number when you rely on "
        "them. If the context does not contain the answer, say so plainly "
        "rather than guessing."
    )

    async def answer(self, context: StrategyContext) -> str:
        retrieved_text = format_retrieved_context(context.retrieved)
        if retrieved_text:
            user = f"Context:\n{retrieved_text}\n\nQuestion: {context.query}"
        else:
            user = (
                f"Question: {context.query}\n\n"
                "No relevant context was retrieved. Answer only if it is general "
                "knowledge; otherwise say you do not know."
            )
        return await self._llm.complete(self.SYSTEM_PROMPT, user, self._max_tokens)
