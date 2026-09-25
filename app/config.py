from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "regulatory-rag"
    environment: str = "local"
    log_level: str = "INFO"

    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/regrag"
    redis_url: str = "redis://localhost:6379/0"

    # Networks that terminate TLS with their own root CA need that bundle passed explicitly.
    ca_bundle: str | None = None

    llm_api_base: str = "https://integrate.api.nvidia.com/v1"
    llm_api_key: str = ""
    llm_model: str = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
    embedding_model: str = "nvidia/nemotron-3-embed-1b"
    embedding_dimensions: int = 2048
    embedding_batch_size: int = 32
    request_timeout: float = 120.0
    upstream_max_attempts: int = 6
    retrieval_top_k: int = 5

    # Nemotron models read this as a control directive and skip extended reasoning, which cuts
    # answer latency roughly sevenfold. Set it to an empty string for providers that ignore it.
    llm_system_prefix: str = "detailed thinking off"


@lru_cache
def get_settings() -> Settings:
    return Settings()
