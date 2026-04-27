from src.known_gap.application.strategies.base import (
    AskStrategy,
    StrategyContext,
    format_known_concepts,
    format_retrieved_context,
    format_unknown_concepts,
)


class ConciseStrategy(AskStrategy):
    SYSTEM_TEMPLATE = (
        "You are a helpful assistant in CONCISE mode. Answer the question "
        "using the retrieved context, citing sources by their [index] number.\n\n"
        "The user ALREADY KNOWS these concepts: {known_list}\n"
        "These concepts are NEW: {unknown_list}\n\n"
        "Do NOT re-explain anything in the known list. Assume the user "
        "understands those concepts and focus only on what is NEW to them. "
        "Be brief.\n\n"
        "You have access to graph tools but should only use them when "
        "absolutely necessary — concise mode prioritises speed. If the "
        "context does not contain the answer, say so plainly."
    )

    async def answer(self, context: StrategyContext) -> str:
        system = self.SYSTEM_TEMPLATE.format(
            known_list=format_known_concepts(context.known),
            unknown_list=format_unknown_concepts(context.unknown),
        )
        retrieved_text = format_retrieved_context(context.retrieved)
        if retrieved_text:
            user = f"Context:\n{retrieved_text}\n\nQuestion: {context.query}"
        else:
            user = f"Question: {context.query}\n\nNo relevant context was retrieved."
        return await self._answerer.answer(system, user, context.tools)
