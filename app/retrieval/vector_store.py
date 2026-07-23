"""Persistent Chroma storage and similarity search for document chunks."""

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

DEFAULT_TOP_K = 4


class DocumentVectorStore:
    """Store, replace, filter, and search traceable document chunks."""

    def __init__(
        self,
        *,
        embeddings: Embeddings,
        persist_directory: str | Path,
        collection_name: str = "documentation",
    ) -> None:
        self._persist_directory = Path(persist_directory).expanduser().resolve()
        self._persist_directory.mkdir(parents=True, exist_ok=True)
        self._store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(self._persist_directory),
        )

    def add_chunks(self, chunks: Iterable[Document]) -> list[str]:
        """Replace indexed documents represented by chunks and return chunk IDs."""
        chunk_list = list(chunks)
        if not chunk_list:
            return []

        chunk_ids: list[str] = []
        document_ids: set[str] = set()
        for chunk in chunk_list:
            document_id = _required_metadata_string(chunk, "document_id")
            chunk_id = _required_metadata_string(chunk, "chunk_id")
            if not chunk.page_content.strip():
                raise ValueError(f"Chunk '{chunk_id}' must not be empty.")
            document_ids.add(document_id)
            chunk_ids.append(chunk_id)

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("Chunk IDs must be unique within an indexing batch.")

        for document_id in sorted(document_ids):
            self.delete_document(document_id)
        return self._store.add_documents(documents=chunk_list, ids=chunk_ids)

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        document_ids: Sequence[str] | None = None,
    ) -> list[Document]:
        """Return chunks nearest to a query, optionally restricted by document ID."""
        if not query.strip():
            raise ValueError("Search query must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        search_filter: dict[str, Any] | None = None
        if document_ids is not None:
            normalized_ids = list(dict.fromkeys(document_ids))
            if not normalized_ids:
                return []
            if any(not document_id.strip() for document_id in normalized_ids):
                raise ValueError("document_ids must not contain empty values.")
            search_filter = {"document_id": {"$in": normalized_ids}}

        return self._store.similarity_search(
            query=query,
            k=top_k,
            filter=search_filter,
        )

    def delete_document(self, document_id: str) -> None:
        """Delete every indexed chunk belonging to one document."""
        if not document_id.strip():
            raise ValueError("document_id must not be empty.")
        self._store.delete(where={"document_id": document_id})


def _required_metadata_string(chunk: Document, key: str) -> str:
    """Read a required non-empty string from chunk metadata."""
    value = chunk.metadata.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Every chunk must have a non-empty {key}.")
    return value
