from src.known_gap.application.services.prompt_builder import PromptBuilder
from src.known_gap.application.use_cases.ask.command import AskCommand
from src.known_gap.application.use_cases.ask.result import AskResult, AskSource
from src.known_gap.domain.repositories.chunk_repository import ChunkRepository
from src.known_gap.domain.services.embedding_provider import EmbeddingProvider
from src.known_gap.domain.services.llm_provider import LLMProvider
from src.known_gap.shared.exceptions.base import PermanentException

PREVIEW_CHARS = 200


class AskHandler:
    def __init__(
        self,
        embedder: EmbeddingProvider,
        chunks: ChunkRepository,
        llm: LLMProvider,
        prompt_builder: PromptBuilder,
        top_k: int,
        answer_max_tokens: int,
    ) -> None:
        self._embedder = embedder
        self._chunks = chunks
        self._llm = llm
        self._prompt_builder = prompt_builder
        self._top_k = top_k
        self._answer_max_tokens = answer_max_tokens

    async def execute(self, command: AskCommand) -> AskResult:
        if command.mode != "normal":
            raise PermanentException(
                message=f"Mode '{command.mode}' is not implemented yet",
                error_code="MODE_NOT_IMPLEMENTED",
                details={"requested_mode": command.mode, "supported": ["normal"]},
            )

        query_embedding = await self._embedder.embed_query(command.query)
        retrieved = await self._chunks.search_similar(query_embedding, self._top_k)
        system, user = self._prompt_builder.build(command.query, retrieved)
        answer = await self._llm.complete(system, user, self._answer_max_tokens)

        sources = [
            AskSource(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                filename=chunk.document_filename,
                content_preview=chunk.content[:PREVIEW_CHARS],
                similarity_score=chunk.similarity_score,
            )
            for chunk in retrieved
        ]
        return AskResult(answer=answer, sources=sources, mode=command.mode)
