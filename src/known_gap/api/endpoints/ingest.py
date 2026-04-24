from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.known_gap.api.dependencies import CurrentUserId, IngestHandlerDep, SettingsDep
from src.known_gap.api.schemas.ingest import IngestResponse
from src.known_gap.application.use_cases.ingest_document.command import IngestDocumentCommand
from src.known_gap.shared.exceptions.base import AppException

router = APIRouter()

SUPPORTED_SUFFIXES = frozenset({".pdf", ".md", ".txt"})

UploadedFile = Annotated[UploadFile, File(...)]


@router.post("/ingest", response_model=IngestResponse, status_code=201)
async def ingest(
    handler: IngestHandlerDep,
    settings: SettingsDep,
    user_id: CurrentUserId,  # noqa: ARG001 — auth gate; documents are global
    file: UploadedFile,
) -> IngestResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=(f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_SUFFIXES)}"),
        )

    data = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds max upload size of {settings.max_upload_mb} MB",
        )

    command = IngestDocumentCommand(
        filename=file.filename,
        content_type=file.content_type or "application/octet-stream",
        data=data,
    )
    try:
        result = await handler.execute(command)
    except AppException as e:
        raise HTTPException(status_code=e.http_status_code, detail=e.message) from e

    return IngestResponse(
        document_id=result.document_id,
        filename=result.filename,
        chunk_count=result.chunk_count,
    )
