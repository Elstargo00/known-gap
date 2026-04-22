"""Entry point."""

import uvicorn
from fastapi import FastAPI

from src.known_gap.config.settings import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.project_name,
    version=settings.project_version,
    debug=settings.debug,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=settings.debug)
