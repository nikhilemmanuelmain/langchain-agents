"""Replaceable embedding-provider construction."""

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from app.config import Settings


class EmbeddingConfigurationError(ValueError):
    """Raised when real embeddings cannot be configured safely."""


def create_embeddings(settings: Settings) -> Embeddings:
    """Create the configured OpenAI embedding implementation."""
    if (
        settings.openai_api_key is None
        or not settings.openai_api_key.get_secret_value()
    ):
        raise EmbeddingConfigurationError(
            "OPENAI_API_KEY is required to create OpenAI embeddings."
        )
    if not settings.openai_embedding_model.strip():
        raise EmbeddingConfigurationError("OPENAI_EMBEDDING_MODEL must not be empty.")
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
        timeout=settings.openai_request_timeout_seconds,
        max_retries=settings.openai_max_retries,
    )
