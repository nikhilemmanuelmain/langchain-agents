"""Command-line utility for inspecting loaded and split documents."""

import argparse
import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from app.config import get_settings
from app.ingestion.loaders import DocumentLoadError, load_documents
from app.ingestion.splitter import split_documents


def _parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Load supported files, split them, and print chunk metadata."
    )
    parser.add_argument("paths", nargs="+", type=Path, help="Files to chunk")
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    parser.add_argument(
        "--preview-length",
        type=int,
        default=300,
        help="Maximum content characters printed per chunk (default: 300)",
    )
    return parser.parse_args()


def _serialize_chunk(chunk: Document, preview_length: int) -> dict[str, Any]:
    return {
        "metadata": chunk.metadata,
        "content_preview": chunk.page_content[:preview_length],
        "content_length": len(chunk.page_content),
    }


def main() -> None:
    """Load, split, and print chunks for command-line paths."""
    args = _parse_args()
    if args.preview_length < 0:
        raise SystemExit("--preview-length must be zero or greater.")

    try:
        documents = load_documents(args.paths)
        chunks = split_documents(
            documents,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    except (DocumentLoadError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    output = [_serialize_chunk(chunk, args.preview_length) for chunk in chunks]
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
