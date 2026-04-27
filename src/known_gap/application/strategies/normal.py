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
        "rather than guessing.\n\n"
        "You have read-only access to the user's personal knowledge graph via "
        "tools. You MAY consult the graph to ground your answer in what the "
        "user already knows, but for normal mode this is optional — only use "
        "tools if the question would clearly benefit from cross-concept "
        "context (e.g. when the user references something you'd want to "
        "verify they understand)."
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
        return await self._answerer.answer(self.SYSTEM_PROMPT, user, context.tools)
