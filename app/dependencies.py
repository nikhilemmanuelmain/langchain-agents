"""FastAPI dependency construction for replaceable application services."""

from functools import lru_cache

from fastapi import HTTPException

from app.config import get_settings
from app.conversations.service import ConversationService
from app.conversations.store import ConversationStore
from app.generation.chat_model import (
    ChatModelConfigurationError,
    create_answer_chain,
    create_chat_model,
    create_rewrite_chain,
)
from app.generation.rag_service import RagService
from app.retrieval.embeddings import EmbeddingConfigurationError, create_embeddings
from app.retrieval.vector_store import DocumentVectorStore
from app.services.document_service import DocumentService


@lru_cache
def _get_vector_store() -> DocumentVectorStore:
    settings = get_settings()
    return DocumentVectorStore(
        embeddings=create_embeddings(settings),
        persist_directory=settings.chroma_persist_directory,
        collection_name=settings.chroma_collection_name,
    )


@lru_cache
def _build_rag_service() -> RagService:
    settings = get_settings()
    retriever = _get_vector_store()
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


@lru_cache
def _build_conversation_service() -> ConversationService:
    settings = get_settings()
    chat_model = create_chat_model(settings)
    return ConversationService(
        rag_service=_build_rag_service(),
        rewrite_chain=create_rewrite_chain(chat_model),
        store=ConversationStore(
            max_turns=settings.max_conversation_turns,
            max_conversations=settings.max_conversations,
        ),
    )


def get_conversation_service() -> ConversationService:
    """Return configured conversational RAG or a safe availability error."""
    try:
        return _build_conversation_service()
    except (EmbeddingConfigurationError, ChatModelConfigurationError) as exc:
        raise HTTPException(
            status_code=503,
            detail="The chat service is not configured.",
        ) from exc


@lru_cache
def get_document_service() -> DocumentService:
    """Return document management with lazy vector-store construction."""
    settings = get_settings()
    return DocumentService(
        documents_directory=settings.documents_directory,
        registry_path=settings.document_registry_path,
        vector_store_provider=_get_vector_store,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        max_document_size_bytes=settings.max_document_size_bytes,
    )
