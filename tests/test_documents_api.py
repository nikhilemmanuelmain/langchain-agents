"""API tests for document upload and management routes."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.dependencies import get_document_service
from app.main import app
from app.services.document_service import DocumentService

client = TestClient(app)


class ApiVectorStore:
    def __init__(self) -> None:
        self.documents: list[Document] = []
        self.deleted: list[str] = []

    def add_chunks(self, chunks: list[Document]) -> list[str]:
        self.documents = chunks
        return [str(chunk.metadata["chunk_id"]) for chunk in chunks]

    def delete_document(self, document_id: str) -> None:
        self.deleted.append(document_id)


@pytest.fixture(autouse=True)
def override_document_service(tmp_path: Path) -> DocumentService:
    vector_store = ApiVectorStore()
    service = DocumentService(
        documents_directory=tmp_path / "documents",
        registry_path=tmp_path / "registry.json",
        vector_store_provider=lambda: vector_store,  # type: ignore[arg-type]
        chunk_size=80,
        chunk_overlap=10,
        max_document_size_bytes=128,
    )
    app.dependency_overrides[get_document_service] = lambda: service
    yield service
    app.dependency_overrides.clear()


def test_document_api_lifecycle() -> None:
    upload = client.post(
        "/api/documents",
        files={"file": ("guide.md", b"Reset passwords in Account Settings.")},
    )

    assert upload.status_code == 201
    record = upload.json()
    assert record["status"] == "indexed"
    assert "source" not in record
    document_id = record["document_id"]

    duplicate = client.post(
        "/api/documents",
        files={"file": ("guide.md", b"Reset passwords in Account Settings.")},
    )
    assert duplicate.status_code == 200

    listed = client.get("/api/documents")
    assert listed.status_code == 200
    assert listed.json() == [record]

    fetched = client.get(f"/api/documents/{document_id}")
    assert fetched.status_code == 200
    assert fetched.json() == record

    reindexed = client.post(f"/api/documents/{document_id}/reindex")
    assert reindexed.status_code == 200
    assert reindexed.json()["status"] == "indexed"

    deleted = client.delete(f"/api/documents/{document_id}")
    assert deleted.status_code == 204
    assert client.get(f"/api/documents/{document_id}").status_code == 404


def test_document_api_rejects_unsupported_file() -> None:
    response = client.post(
        "/api/documents",
        files={"file": ("data.csv", b"a,b")},
    )

    assert response.status_code == 415
    assert "Unsupported document type" in response.json()["detail"]


def test_document_api_rejects_oversized_file() -> None:
    response = client.post(
        "/api/documents",
        files={"file": ("large.txt", b"x" * 129)},
    )

    assert response.status_code == 413
