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

    graph_expert_model: str = "claude-opus-4-7"
    graph_expert_max_tokens: int = 2048
    graph_expert_degree: int = 3

    known_score_threshold: int = 50
    known_score_increment: int = 10
    known_score_decrement: int = 10
    # First-exposure score for concepts the user has actually been shown
    # (answer-introduced concepts and ingested documents). Must be
    # *strictly above* `known_score_threshold` so that a topic-of-the-
    # question concept (which gets neither +increment nor -decrement
    # because it appears in BOTH question and answer) is classified as
    # known on the very next turn — otherwise it parks at the initial
    # value forever and is never cloze-masked. One decrement of slack
    # also lets a re-ask gracefully drop the concept back below the bar
    # ("you just had to look it up — not yet mastered"). LLM-proposed
    # graph-expander neighbours still seed at 0 (see GraphExpander)
    # because the user hasn't actually seen them.
    known_score_initial: int = 60

    concept_neighborhood_max_hops: int = 2
    tool_max_iterations: int = 6

    falkordb_host: str = "localhost"
    falkordb_port: int = 6379
    falkordb_username: str = ""
    falkordb_password: str = ""

    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"


@lru_cache
def get_settings() -> Settings:
    return Settings()
