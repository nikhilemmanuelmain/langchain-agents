"""Manual end-to-end indexing and grounded-answer command."""

import argparse
import json
from pathlib import Path

from app.config import get_settings
from app.generation.chat_model import create_answer_chain, create_chat_model
from app.generation.rag_service import RagService, RagServiceUnavailableError
from app.ingestion.loaders import DocumentLoadError, load_documents
from app.ingestion.splitter import split_documents
from app.retrieval.embeddings import EmbeddingConfigurationError, create_embeddings
from app.retrieval.vector_store import DocumentVectorStore


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load, index, retrieve, and answer from documentation."
    )
    parser.add_argument("question", help="Documentation question")
    parser.add_argument("paths", nargs="+", type=Path, help="Files to index")
    return parser.parse_args()


def main() -> None:
    """Run the complete Module 5 pipeline for manual verification."""
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
        service = RagService(
            retriever=vector_store,
            answer_chain=create_answer_chain(create_chat_model(settings)),
            top_k=settings.retrieval_top_k,
        )
        response = service.answer(args.question)
    except (
        DocumentLoadError,
        EmbeddingConfigurationError,
        RagServiceUnavailableError,
        ValueError,
    ) as exc:
        raise SystemExit(str(exc)) from exc

    print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
