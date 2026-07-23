"""Opt-in real OpenAI end-to-end test for the Module 5 RAG pipeline."""

import os
from pathlib import Path

import pytest

from app.config import Settings
from app.generation.chat_model import create_answer_chain, create_chat_model
from app.generation.rag_service import RagService
from app.ingestion.loaders import load_document
from app.ingestion.splitter import split_documents
from app.retrieval.embeddings import create_embeddings
from app.retrieval.vector_store import DocumentVectorStore

pytestmark = pytest.mark.integration

if os.environ.get("RUN_OPENAI_INTEGRATION") != "1":
    pytest.skip(
        "Set RUN_OPENAI_INTEGRATION=1 to run paid OpenAI integration tests.",
        allow_module_level=True,
    )


def test_real_grounded_answer_pipeline(tmp_path: Path) -> None:
    """Load, chunk, embed, retrieve, and answer from the test guide."""
    settings = Settings()
    fixture = Path(__file__).parents[1] / "fixtures" / "documents" / "guide.md"
    chunks = split_documents(load_document(fixture))
    vector_store = DocumentVectorStore(
        embeddings=create_embeddings(settings),
        persist_directory=tmp_path / "chroma",
        collection_name="manual-rag-e2e",
    )
    vector_store.add_chunks(chunks)
    service = RagService(
        retriever=vector_store,
        answer_chain=create_answer_chain(create_chat_model(settings)),
        top_k=settings.retrieval_top_k,
    )

    response = service.answer("Where can I reset my password?")

    assert "Account Settings" in response.answer
    assert response.sources
    assert response.sources[0].filename == "guide.md"
