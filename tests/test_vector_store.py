"""Tests for embedding configuration and persistent Chroma vector search."""

from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config import Settings
from app.retrieval.embeddings import (
    EmbeddingConfigurationError,
    create_embeddings,
)
from app.retrieval.vector_store import DocumentVectorStore


class KeywordEmbeddings(Embeddings):
    """Deterministic local embeddings for isolated retrieval tests."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        normalized = text.lower()
        return [
            float("password" in normalized or "account" in normalized),
            float("leave" in normalized or "vacation" in normalized),
            float("security" in normalized),
        ]


def make_chunk(
    content: str,
    *,
    document_id: str,
    chunk_index: int = 0,
) -> Document:
    """Create a traceable chunk for vector-store tests."""
    return Document(
        page_content=content,
        metadata={
            "document_id": document_id,
            "filename": f"{document_id}.md",
            "source": f"data/documents/{document_id}.md",
            "file_type": "markdown",
            "chunk_index": chunk_index,
            "chunk_id": f"{document_id}-chunk-{chunk_index}",
        },
    )


def make_store(path: Path) -> DocumentVectorStore:
    """Create an isolated persistent store with fake embeddings."""
    return DocumentVectorStore(
        embeddings=KeywordEmbeddings(),
        persist_directory=path,
        collection_name="test-documentation",
    )


def test_search_returns_semantically_matching_chunk(tmp_path: Path) -> None:
    store = make_store(tmp_path / "chroma")
    password = make_chunk(
        "Reset your password in Account Settings.", document_id="guide"
    )
    leave = make_chunk("Employees receive annual leave.", document_id="policy")
    store.add_chunks([password, leave])

    results = store.search("How do I change my password?", top_k=1)

    assert len(results) == 1
    assert results[0].metadata["document_id"] == "guide"


def test_search_filters_by_document_ids(tmp_path: Path) -> None:
    store = make_store(tmp_path / "chroma")
    store.add_chunks(
        [
            make_chunk("Account password instructions.", document_id="guide"),
            make_chunk("Account security policy.", document_id="security"),
        ]
    )

    results = store.search(
        "account password",
        top_k=4,
        document_ids=["security"],
    )

    assert [result.metadata["document_id"] for result in results] == ["security"]


def test_empty_document_filter_returns_no_results(tmp_path: Path) -> None:
    store = make_store(tmp_path / "chroma")
    store.add_chunks([make_chunk("Password help.", document_id="guide")])

    assert store.search("password", document_ids=[]) == []


def test_reindex_replaces_all_existing_document_chunks(tmp_path: Path) -> None:
    store = make_store(tmp_path / "chroma")
    store.add_chunks(
        [
            make_chunk("Old password text.", document_id="guide", chunk_index=0),
            make_chunk("Old account text.", document_id="guide", chunk_index=1),
        ]
    )

    store.add_chunks(
        [make_chunk("New password instructions.", document_id="guide", chunk_index=0)]
    )
    results = store.search("password", top_k=10, document_ids=["guide"])

    assert [result.page_content for result in results] == ["New password instructions."]


def test_delete_document_removes_its_chunks_only(tmp_path: Path) -> None:
    store = make_store(tmp_path / "chroma")
    store.add_chunks(
        [
            make_chunk("Password help.", document_id="guide"),
            make_chunk("Vacation policy.", document_id="policy"),
        ]
    )

    store.delete_document("guide")

    assert store.search("password", document_ids=["guide"]) == []
    remaining = store.search("vacation", document_ids=["policy"])
    assert [result.metadata["document_id"] for result in remaining] == ["policy"]


def test_data_persists_between_store_instances(tmp_path: Path) -> None:
    persist_directory = tmp_path / "chroma"
    first_store = make_store(persist_directory)
    first_store.add_chunks([make_chunk("Vacation policy.", document_id="policy")])

    second_store = make_store(persist_directory)
    results = second_store.search("annual leave", top_k=1)

    assert results[0].metadata["document_id"] == "policy"


def test_add_chunks_validates_traceability_metadata(tmp_path: Path) -> None:
    store = make_store(tmp_path / "chroma")
    chunk = Document(page_content="Content", metadata={"document_id": "guide"})

    with pytest.raises(ValueError, match="chunk_id"):
        store.add_chunks([chunk])


def test_duplicate_chunk_ids_are_rejected(tmp_path: Path) -> None:
    store = make_store(tmp_path / "chroma")
    first = make_chunk("First", document_id="guide")
    second = make_chunk("Second", document_id="guide")

    with pytest.raises(ValueError, match="unique"):
        store.add_chunks([first, second])


@pytest.mark.parametrize(("query", "top_k"), [("", 4), ("question", 0)])
def test_search_validates_input(tmp_path: Path, query: str, top_k: int) -> None:
    store = make_store(tmp_path / "chroma")

    with pytest.raises(ValueError):
        store.search(query, top_k=top_k)


def test_openai_embeddings_require_api_key() -> None:
    with pytest.raises(EmbeddingConfigurationError, match="OPENAI_API_KEY"):
        create_embeddings(Settings(openai_api_key=None))


def test_openai_embedding_configuration_is_replaceable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    fake_embeddings = KeywordEmbeddings()

    def fake_factory(**kwargs: object) -> Embeddings:
        captured.update(kwargs)
        return fake_embeddings

    monkeypatch.setattr(
        "app.retrieval.embeddings.OpenAIEmbeddings",
        fake_factory,
    )
    settings = Settings(
        openai_api_key="test-key",
        openai_embedding_model="test-embedding-model",
    )

    result = create_embeddings(settings)

    assert result is fake_embeddings
    assert captured["model"] == "test-embedding-model"
