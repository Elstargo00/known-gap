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

    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    llm_primary_model: str = "claude-sonnet-4-6"
    llm_fallback_model: str = "gemini-2.5-flash"
    answer_max_tokens: int = 1024

    top_k: int = 10

    concept_extraction_model: str = "claude-haiku-4-5-20251001"
    concept_extraction_max_tokens: int = 1024

    falkordb_host: str = "localhost"
    falkordb_port: int = 6379
    falkordb_username: str = ""
    falkordb_password: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
