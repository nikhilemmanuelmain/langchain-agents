"""Unit tests for document loading and metadata preservation."""

from pathlib import Path

import pytest
from langchain_core.documents import Document
from pypdf import PdfWriter

from app.ingestion.loaders import (
    DocumentLoadError,
    UnsupportedFileTypeError,
    load_document,
    load_documents,
)

FIXTURES = Path(__file__).parent / "fixtures" / "documents"


def test_load_markdown_document_with_metadata() -> None:
    documents = load_document(FIXTURES / "guide.md")

    assert len(documents) == 1
    document = documents[0]
    assert isinstance(document, Document)
    assert "Reset your password" in document.page_content
    assert document.metadata == {
        "document_id": document.metadata["document_id"],
        "filename": "guide.md",
        "source": "tests/fixtures/documents/guide.md",
        "file_type": "markdown",
    }
    assert document.metadata["document_id"].startswith("guide-")


def test_load_text_document() -> None:
    document = load_document(FIXTURES / "policy.txt")[0]

    assert document.page_content == "Employees receive 26 annual leave days.\n"
    assert document.metadata["file_type"] == "text"
    assert "page" not in document.metadata


def test_document_id_is_stable() -> None:
    first_id = load_document(FIXTURES / "guide.md")[0].metadata["document_id"]
    second_id = load_document(FIXTURES / "guide.md")[0].metadata["document_id"]

    assert first_id == second_id


def test_document_id_changes_when_content_changes(tmp_path: Path) -> None:
    document_path = tmp_path / "guide.md"
    document_path.write_text("First version", encoding="utf-8")
    first_id = load_document(document_path)[0].metadata["document_id"]
    document_path.write_text("Second version", encoding="utf-8")

    second_id = load_document(document_path)[0].metadata["document_id"]

    assert first_id != second_id


def test_load_pdf_as_one_document_per_page(tmp_path: Path) -> None:
    pdf_path = tmp_path / "handbook.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=792)
    with pdf_path.open("wb") as file_handle:
        writer.write(file_handle)

    documents = load_document(pdf_path)

    assert len(documents) == 2
    assert [document.metadata["page"] for document in documents] == [1, 2]
    assert all(
        document.metadata["filename"] == "handbook.pdf" for document in documents
    )
    assert all(document.metadata["file_type"] == "pdf" for document in documents)
    assert documents[0].metadata["document_id"] == documents[1].metadata["document_id"]


def test_load_multiple_files_preserves_order() -> None:
    documents = load_documents([FIXTURES / "policy.txt", FIXTURES / "guide.md"])

    assert [document.metadata["filename"] for document in documents] == [
        "policy.txt",
        "guide.md",
    ]


def test_reject_unsupported_file_type() -> None:
    with pytest.raises(UnsupportedFileTypeError, match="Unsupported document type"):
        load_document(FIXTURES / "unsupported.csv")


def test_reject_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DocumentLoadError, match="does not exist"):
        load_document(tmp_path / "missing.md")


def test_reject_directory_path(tmp_path: Path) -> None:
    with pytest.raises(DocumentLoadError, match="not a file"):
        load_document(tmp_path)


def test_reject_malformed_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "broken.pdf"
    pdf_path.write_text("This is not a PDF.", encoding="utf-8")

    with pytest.raises(DocumentLoadError, match="Could not read PDF"):
        load_document(pdf_path)
