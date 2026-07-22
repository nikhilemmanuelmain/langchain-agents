"""Command-line inspection utility for loaded document content and metadata."""

import argparse
import json
from pathlib import Path
from typing import Any

from app.ingestion.loaders import DocumentLoadError, load_documents


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load supported files and print their content and metadata."
    )
    parser.add_argument("paths", nargs="+", type=Path, help="Files to inspect")
    parser.add_argument(
        "--preview-length",
        type=int,
        default=300,
        help="Maximum content characters printed per document (default: 300)",
    )
    return parser.parse_args()


def _serialize_document(document: Any, preview_length: int) -> dict[str, Any]:
    return {
        "metadata": document.metadata,
        "content_preview": document.page_content[:preview_length],
        "content_length": len(document.page_content),
    }


def main() -> None:
    """Load command-line paths and print inspectable JSON."""
    args = _parse_args()
    if args.preview_length < 0:
        raise SystemExit("--preview-length must be zero or greater.")

    try:
        documents = load_documents(args.paths)
    except DocumentLoadError as exc:
        raise SystemExit(str(exc)) from exc

    output = [
        _serialize_document(document, args.preview_length) for document in documents
    ]
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
