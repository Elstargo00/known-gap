from fastapi import APIRouter, HTTPException

from src.known_gap.api.dependencies import AskHandlerDep
from src.known_gap.api.schemas.ask import AskRequest, AskResponse, Source
from src.known_gap.application.use_cases.ask.command import AskCommand
from src.known_gap.shared.exceptions.base import AppException

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask(
    request: AskRequest,
    handler: AskHandlerDep,
) -> AskResponse:
    command = AskCommand(query=request.query, mode=request.mode)
    try:
        result = await handler.execute(command)
    except AppException as e:
        raise HTTPException(status_code=e.http_status_code, detail=e.message) from e

    return AskResponse(
        answer=result.answer,
        sources=[
            Source(
                chunk_id=source.chunk_id,
                document_id=source.document_id,
                filename=source.filename,
                content_preview=source.content_preview,
                similarity_score=source.similarity_score,
            )
            for source in result.sources
        ],
        mode=result.mode,
    )
