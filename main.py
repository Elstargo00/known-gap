"""Entry point."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from src.known_gap.api.endpoints.ask import router as ask_router
from src.known_gap.api.endpoints.ingest import router as ingest_router
from src.known_gap.config.settings import get_settings
from src.known_gap.infrastructure.db.pool import create_pool
from src.known_gap.infrastructure.graph.falkordb_client import create_falkordb_client

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.pool = await create_pool(settings.database_url)
    app.state.graph_client = create_falkordb_client(settings)
    try:
        yield
    finally:
        await app.state.pool.close()


app = FastAPI(
    title=settings.project_name,
    version=settings.project_version,
    debug=settings.debug,
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(ingest_router)
app.include_router(ask_router)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=settings.debug)
