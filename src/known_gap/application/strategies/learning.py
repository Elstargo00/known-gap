from src.known_gap.application.strategies.base import (
    AskStrategy,
    StrategyContext,
    format_known_concepts,
    format_retrieved_context,
)


class LearningStrategy(AskStrategy):
    SYSTEM_TEMPLATE = (
        "You are a helpful assistant in LEARNING mode. Answer the question using "
        "the retrieved context, citing sources by their [index] number.\n\n"
        "The user ALREADY KNOWS these concepts: {known_list}\n\n"
        "Whenever your answer mentions or explains any of those known concepts, "
        'wrap the relevant passage in <known concept="<name>">...</known> tags '
        "so the UI can visually de-emphasise them. Keep the answer itself natural "
        "and complete — do not omit known material, just tag it. If the context "
        "does not contain the answer, say so plainly."
    )

    async def answer(self, context: StrategyContext) -> str:
        system = self.SYSTEM_TEMPLATE.format(known_list=format_known_concepts(context.known))
        retrieved_text = format_retrieved_context(context.retrieved)
        if retrieved_text:
            user = f"Context:\n{retrieved_text}\n\nQuestion: {context.query}"
        else:
            user = f"Question: {context.query}\n\nNo relevant context was retrieved."
        return await self._llm.complete(system, user, self._max_tokens)
