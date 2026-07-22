"""Split loaded documents into traceable chunks."""

from collections.abc import Iterable
from hashlib import sha256
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120


def split_documents(
    documents: Iterable[Document],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents while preserving metadata and adding stable chunk IDs."""
    _validate_chunk_settings(chunk_size, chunk_overlap)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    chunks: list[Document] = []
    for document in documents:
        if not document.page_content.strip():
            continue

        document_id = document.metadata.get("document_id")
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("Every document must have a non-empty document_id.")

        split_content = splitter.split_text(document.page_content)
        for chunk_index, content in enumerate(split_content):
            if not content.strip():
                continue
            metadata = {
                **document.metadata,
                "chunk_index": chunk_index,
                "chunk_id": _create_chunk_id(
                    document_id=document_id,
                    page=document.metadata.get("page"),
                    chunk_index=chunk_index,
                    content=content,
                ),
            }
            chunks.append(Document(page_content=content, metadata=metadata))
    return chunks


def _validate_chunk_settings(chunk_size: int, chunk_overlap: int) -> None:
    """Reject settings that cannot produce valid chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be zero or greater.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")


def _create_chunk_id(
    *,
    document_id: str,
    page: Any,
    chunk_index: int,
    content: str,
) -> str:
    """Create a deterministic ID tied to a chunk's source and content."""
    page_component = f"-page-{page}" if page is not None else ""
    content_digest = sha256(content.encode("utf-8")).hexdigest()[:12]
    return f"{document_id}{page_component}-chunk-{chunk_index}-{content_digest}"
