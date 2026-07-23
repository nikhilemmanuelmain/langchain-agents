"""FastAPI dependency construction for replaceable application services."""

from functools import lru_cache

from fastapi import HTTPException

from app.config import get_settings
from app.generation.chat_model import (
    ChatModelConfigurationError,
    create_answer_chain,
    create_chat_model,
)
from app.generation.rag_service import RagService
from app.retrieval.embeddings import EmbeddingConfigurationError, create_embeddings
from app.retrieval.vector_store import DocumentVectorStore


@lru_cache
def _build_rag_service() -> RagService:
    settings = get_settings()
    retriever = DocumentVectorStore(
        embeddings=create_embeddings(settings),
        persist_directory=settings.chroma_persist_directory,
        collection_name=settings.chroma_collection_name,
    )
    answer_chain = create_answer_chain(create_chat_model(settings))
    return RagService(
        retriever=retriever,
        answer_chain=answer_chain,
        top_k=settings.retrieval_top_k,
    )


def get_rag_service() -> RagService:
    """Return the configured RAG service or a safe availability error."""
    try:
        return _build_rag_service()
    except (EmbeddingConfigurationError, ChatModelConfigurationError) as exc:
        raise HTTPException(
            status_code=503,
            detail="The chat service is not configured.",
        ) from exc
