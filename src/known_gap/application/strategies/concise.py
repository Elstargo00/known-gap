from src.known_gap.application.strategies.base import (
    AskStrategy,
    StrategyContext,
    format_known_concepts,
    format_retrieved_context,
)


class ConciseStrategy(AskStrategy):
    SYSTEM_TEMPLATE = (
        "You are a helpful assistant in CONCISE mode. Answer the question using "
        "the retrieved context, citing sources by their [index] number.\n\n"
        "The user ALREADY KNOWS these concepts: {known_list}\n\n"
        "Do NOT re-explain anything in that list. Assume the user understands "
        "those concepts and focus only on what is NEW to them. Be brief. "
        "If the context does not contain the answer, say so plainly."
    )

    async def answer(self, context: StrategyContext) -> str:
        system = self.SYSTEM_TEMPLATE.format(known_list=format_known_concepts(context.known))
        retrieved_text = format_retrieved_context(context.retrieved)
        if retrieved_text:
            user = f"Context:\n{retrieved_text}\n\nQuestion: {context.query}"
        else:
            user = f"Question: {context.query}\n\nNo relevant context was retrieved."
        return await self._llm.complete(system, user, self._max_tokens)
