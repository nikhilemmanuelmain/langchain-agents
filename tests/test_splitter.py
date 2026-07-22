"""Unit tests for document chunking."""

from itertools import pairwise

import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from app.config import Settings
from app.ingestion.splitter import split_documents


def make_document(content: str, **metadata: object) -> Document:
    """Create a loaded-document-shaped test value."""
    base_metadata = {
        "document_id": "guide-stable-id",
        "filename": "guide.pdf",
        "source": "data/documents/guide.pdf",
        "file_type": "pdf",
        "page": 12,
    }
    return Document(page_content=content, metadata={**base_metadata, **metadata})


def test_chunks_do_not_exceed_configured_size() -> None:
    document = make_document("abcdefghijklmnopqrstuvwxyz" * 20)

    chunks = split_documents([document], chunk_size=80, chunk_overlap=10)

    assert len(chunks) > 1
    assert all(0 < len(chunk.page_content) <= 80 for chunk in chunks)


def test_chunks_preserve_metadata_and_add_traceability() -> None:
    source = make_document("A long policy sentence. " * 20, section="Leave")
    original_metadata = source.metadata.copy()

    chunks = split_documents([source], chunk_size=70, chunk_overlap=10)

    assert source.metadata == original_metadata
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(
        range(len(chunks))
    )
    for chunk in chunks:
        assert chunk.metadata["document_id"] == "guide-stable-id"
        assert chunk.metadata["filename"] == "guide.pdf"
        assert chunk.metadata["source"] == "data/documents/guide.pdf"
        assert chunk.metadata["page"] == 12
        assert chunk.metadata["section"] == "Leave"
        assert chunk.metadata["chunk_id"].startswith("guide-stable-id-page-12-chunk-")


def test_chunks_include_configured_overlap() -> None:
    document = make_document("abcdefghijklmnopqrstuvwxyz")

    chunks = split_documents([document], chunk_size=10, chunk_overlap=3)

    assert len(chunks) > 1
    for previous, current in pairwise(chunks):
        assert previous.page_content[-3:] == current.page_content[:3]


def test_empty_documents_produce_no_chunks() -> None:
    documents = [make_document(""), make_document(" \n\t ")]

    assert split_documents(documents) == []


def test_chunk_ids_are_stable() -> None:
    document = make_document("Stable content. " * 100)

    first = split_documents([document], chunk_size=100, chunk_overlap=20)
    second = split_documents([document], chunk_size=100, chunk_overlap=20)

    assert [chunk.metadata["chunk_id"] for chunk in first] == [
        chunk.metadata["chunk_id"] for chunk in second
    ]


def test_chunk_ids_differ_between_pdf_pages() -> None:
    content = "The same repeated page content. " * 20
    page_one = make_document(content, page=1)
    page_two = make_document(content, page=2)

    chunks = split_documents([page_one, page_two], chunk_size=100, chunk_overlap=20)

    chunk_ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_document_id_is_required() -> None:
    document = Document(page_content="Useful content", metadata={})

    with pytest.raises(ValueError, match="non-empty document_id"):
        split_documents([document])


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (100, -1), (100, 100), (100, 101)],
)
def test_invalid_splitter_configuration(chunk_size: int, chunk_overlap: int) -> None:
    with pytest.raises(ValueError):
        split_documents(
            [make_document("Content")],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


def test_chunk_settings_are_environment_configurable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "256")
    monkeypatch.setenv("CHUNK_OVERLAP", "32")

    settings = Settings()

    assert settings.chunk_size == 256
    assert settings.chunk_overlap == 32


def test_settings_reject_overlap_equal_to_size() -> None:
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
        Settings(chunk_size=100, chunk_overlap=100)
