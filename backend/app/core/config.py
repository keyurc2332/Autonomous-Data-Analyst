"""Central application settings, loaded from environment / .env."""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["google", "groq", "ollama"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- App ----
    APP_NAME: str = "Autonomous Data Analyst"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = True
    API_V1_PREFIX: str = "/api/v1"

    # ---- LLM ----
    LLM_PROVIDER: LLMProvider = "google"
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_RETRIES: int = 3

    GOOGLE_API_KEY: str | None = None
    GOOGLE_MODEL: str = "gemini-2.5-flash"

    GROQ_API_KEY: str | None = None
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:7b"

    # ---- Observability ----
    LANGSMITH_TRACING: bool = False
    LANGSMITH_API_KEY: str | None = None
    LANGSMITH_PROJECT: str = "autonomous-data-analyst"

    # ---- Postgres ----
    POSTGRES_USER: str = "ada"
    POSTGRES_PASSWORD: str = "ada_dev_password"
    POSTGRES_DB: str = "ada"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432

    # ---- Redis ----
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    # ---- Storage ----
    STORAGE_DIR: Path = Path("/app/storage")
    MAX_UPLOAD_MB: int = Field(default=100, gt=0)

    @computed_field
    @property
    def database_url(self) -> str:
        """Async driver URL, used by the application at runtime."""
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @computed_field
    @property
    def sync_database_url(self) -> str:
        """Sync driver URL. Alembic autogenerate uses this."""
        return (
            f"postgresql+psycopg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @computed_field
    @property
    def redis_url(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    @computed_field
    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_MB * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """Cached accessor. Use this everywhere instead of instantiating Settings."""
    return Settings()


settings = get_settings()
