"""Environment-based application configuration."""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings loaded from environment variables and an optional local .env file."""

    app_name: str = "Documentation Chatbot API"
    app_env: str = "development"
    cors_allowed_origins: str = "http://localhost:3000,http://localhost:5173"
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=120, ge=0)
    openai_api_key: SecretStr | None = None
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-5.6-luna"
    openai_request_timeout_seconds: float = Field(default=30, gt=0)
    openai_max_retries: int = Field(default=2, ge=0)
    chroma_persist_directory: Path = Path("data/chroma")
    chroma_collection_name: str = "documentation"
    retrieval_top_k: int = Field(default=4, gt=0)
    documents_directory: Path = Path("data/documents")
    document_registry_path: Path = Path("data/document-registry.json")
    max_document_size_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    max_conversation_turns: int = Field(default=6, gt=0)
    max_conversations: int = Field(default=1000, gt=0)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cors_origins(self) -> list[str]:
        """Return normalized origins from the comma-separated environment value."""
        return [
            origin.strip()
            for origin in self.cors_allowed_origins.split(",")
            if origin.strip()
        ]

    @model_validator(mode="after")
    def validate_chunk_settings(self) -> Self:
        """Ensure overlap cannot consume an entire chunk."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance for the application process."""
    return Settings()
