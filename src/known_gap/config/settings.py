from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    project_name: str = "known-gap"
    project_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    database_url: str = "postgresql://known_gap:known_gap@localhost:5433/known_gap"

    embedding_provider: str = "voyage"
    voyage_api_key: str = ""
    embedding_model: str = "voyage-3"
    embedding_dimension: int = 1024
    embedding_batch_size: int = 64

    chunk_size: int = 2048
    chunk_overlap: int = 256

    max_upload_mb: int = 20


@lru_cache
def get_settings() -> Settings:
    return Settings()
