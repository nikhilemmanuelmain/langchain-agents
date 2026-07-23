"""Manual document-indexing and similarity-search command."""

import argparse
import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from app.config import get_settings
from app.ingestion.loaders import DocumentLoadError, load_documents
from app.ingestion.splitter import split_documents
from app.retrieval.embeddings import EmbeddingConfigurationError, create_embeddings
from app.retrieval.vector_store import DocumentVectorStore


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load, chunk, index, and search documents with OpenAI and Chroma."
    )
    parser.add_argument("query", help="Natural-language search query")
    parser.add_argument("paths", nargs="+", type=Path, help="Files to index")
    parser.add_argument(
        "--document-id",
        action="append",
        dest="document_ids",
        help="Restrict results to a document ID; may be repeated",
    )
    return parser.parse_args()


def _serialize_result(document: Document) -> dict[str, Any]:
    return {
        "content": document.page_content,
        "metadata": document.metadata,
    }


def main() -> None:
    """Index supplied files and print matching chunks as JSON."""
    args = _parse_args()
    settings = get_settings()
    try:
        documents = load_documents(args.paths)
        chunks = split_documents(
            documents,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        vector_store = DocumentVectorStore(
            embeddings=create_embeddings(settings),
            persist_directory=settings.chroma_persist_directory,
            collection_name=settings.chroma_collection_name,
        )
        vector_store.add_chunks(chunks)
        results = vector_store.search(
            args.query,
            top_k=settings.retrieval_top_k,
            document_ids=args.document_ids,
        )
    except (DocumentLoadError, EmbeddingConfigurationError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(
        json.dumps(
            [_serialize_result(document) for document in results],
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
