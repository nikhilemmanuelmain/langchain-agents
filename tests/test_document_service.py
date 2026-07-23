"""Tests for persistent document management and indexing."""

from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.services.document_service import (
    DocumentNotFoundError,
    DocumentService,
    DocumentServiceUnavailableError,
    InvalidDocumentError,
    sanitize_filename,
)


class FakeVectorStore:
    """Record vector mutations without embeddings or external services."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.added_batches: list[list[Document]] = []
        self.deleted_document_ids: list[str] = []

    def add_chunks(self, chunks: list[Document]) -> list[str]:
        if self.fail:
            raise RuntimeError("vector failure")
        self.added_batches.append(chunks)
        return [str(chunk.metadata["chunk_id"]) for chunk in chunks]

    def delete_document(self, document_id: str) -> None:
        if self.fail:
            raise RuntimeError("vector failure")
        self.deleted_document_ids.append(document_id)


@pytest.fixture
def vector_store() -> FakeVectorStore:
    return FakeVectorStore()


@pytest.fixture
def document_service(tmp_path: Path, vector_store: FakeVectorStore) -> DocumentService:
    return DocumentService(
        documents_directory=tmp_path / "documents",
        registry_path=tmp_path / "registry.json",
        vector_store_provider=lambda: vector_store,  # type: ignore[arg-type]
        chunk_size=80,
        chunk_overlap=10,
        max_document_size_bytes=1024,
    )


def test_upload_indexes_and_persists_public_metadata(
    document_service: DocumentService,
    vector_store: FakeVectorStore,
) -> None:
    record, created = document_service.upload_document(
        "../Employee Guide.md",
        b"Employees receive 26 annual leave days.",
    )

    assert created is True
    assert record.filename == "Employee_Guide.md"
    assert record.status == "indexed"
    assert record.chunk_count == 1
    assert record.indexed_at is not None
    assert vector_store.added_batches[0][0].metadata["document_id"] == (
        record.document_id
    )
    assert document_service.get_document(record.document_id) == record
    assert document_service.list_documents() == [record]
    assert "path" not in record.model_dump()
    assert "source" not in record.model_dump()


def test_unchanged_duplicate_is_not_reindexed(
    document_service: DocumentService,
    vector_store: FakeVectorStore,
) -> None:
    first, first_created = document_service.upload_document("guide.txt", b"Content")
    second, second_created = document_service.upload_document("guide.txt", b"Content")

    assert first_created is True
    assert second_created is False
    assert second == first
    assert len(vector_store.added_batches) == 1


def test_registry_survives_new_service_instance(
    tmp_path: Path, vector_store: FakeVectorStore
) -> None:
    kwargs = {
        "documents_directory": tmp_path / "documents",
        "registry_path": tmp_path / "registry.json",
        "vector_store_provider": lambda: vector_store,
        "chunk_size": 80,
        "chunk_overlap": 10,
        "max_document_size_bytes": 1024,
    }
    first_service = DocumentService(**kwargs)  # type: ignore[arg-type]
    record, _ = first_service.upload_document("guide.txt", b"Content")

    second_service = DocumentService(**kwargs)  # type: ignore[arg-type]

    assert second_service.get_document(record.document_id) == record


def test_reindex_replaces_document_chunks(
    document_service: DocumentService,
    vector_store: FakeVectorStore,
) -> None:
    record, _ = document_service.upload_document("guide.txt", b"Content")

    reindexed = document_service.reindex_document(record.document_id)

    assert reindexed.status == "indexed"
    assert len(vector_store.added_batches) == 2


def test_delete_removes_vectors_file_and_registry(
    document_service: DocumentService,
    vector_store: FakeVectorStore,
) -> None:
    record, _ = document_service.upload_document("guide.txt", b"Content")

    document_service.delete_document(record.document_id)

    assert vector_store.deleted_document_ids == [record.document_id]
    with pytest.raises(DocumentNotFoundError):
        document_service.get_document(record.document_id)


@pytest.mark.parametrize(
    ("filename", "content", "message"),
    [
        ("guide.csv", b"content", "Unsupported"),
        ("guide.txt", b"", "must not be empty"),
        ("guide.txt", b"too large", "exceeds"),
    ],
)
def test_upload_validation(
    document_service: DocumentService,
    filename: str,
    content: bytes,
    message: str,
) -> None:
    if message == "exceeds":
        document_service._max_document_size_bytes = 2

    with pytest.raises(InvalidDocumentError, match=message):
        document_service.upload_document(filename, content)


def test_vector_failure_is_recorded_as_failed(tmp_path: Path) -> None:
    vector_store = FakeVectorStore(fail=True)
    service = DocumentService(
        documents_directory=tmp_path / "documents",
        registry_path=tmp_path / "registry.json",
        vector_store_provider=lambda: vector_store,  # type: ignore[arg-type]
        chunk_size=80,
        chunk_overlap=10,
        max_document_size_bytes=1024,
    )

    with pytest.raises(DocumentServiceUnavailableError):
        service.upload_document("guide.txt", b"Content")

    record = service.list_documents()[0]
    assert record.status == "failed"
    assert record.error == "Document indexing failed."


def test_filename_sanitization_blocks_traversal() -> None:
    assert sanitize_filename("../../private policy.md") == "private_policy.md"
    assert sanitize_filename(r"..\..\windows.txt") == "windows.txt"
