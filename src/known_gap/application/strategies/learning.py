from src.known_gap.application.strategies.base import (
    AskStrategy,
    StrategyContext,
    format_known_concepts,
    format_retrieved_context,
    format_unknown_concepts,
)


class LearningStrategy(AskStrategy):
    """Learning mode produces a *natural* answer that the cloze
    post-processor will then rewrite into fill-in-the-blank exercises
    around concepts the user is already known to "own". The model itself
    does not insert tags — it just writes the most useful answer it can,
    leaning on the graph tools to traverse what the user knows so it can
    relate new ideas to existing ones."""

    SYSTEM_TEMPLATE = (
        "You are a helpful assistant in LEARNING mode. Answer the question "
        "using the retrieved context, citing sources by their [index] number.\n\n"
        "The user ALREADY KNOWS these concepts: {known_list}\n"
        "These concepts are NEW or not yet 'owned' by the user: {unknown_list}\n\n"
        "Your job is to teach. Use the graph tools when helpful: "
        "`get_concept_neighborhood` or `find_path_between_concepts` let you "
        "see how the user's known concepts connect to each other, so you can "
        "explain new ideas by analogy to what they already understand. "
        "`lookup_user_known_concepts` reports the user's score for any "
        "concept (>50 = known).\n\n"
        "Write the answer naturally and completely — do NOT add tags or "
        "markup yourself. The system will post-process the answer to mask "
        "known concepts as cloze fill-in-the-blanks for the user to test "
        "themselves on. If the context does not contain the answer, say so "
        "plainly."
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
