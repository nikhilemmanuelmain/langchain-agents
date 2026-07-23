"""Command-line runner for the checked-in RAG evaluation dataset."""

import argparse
import json
from pathlib import Path

from fastapi import HTTPException

from app.dependencies import get_rag_service
from app.evaluation.evaluator import evaluate_dataset, load_dataset
from app.generation.rag_service import RagServiceUnavailableError


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the indexed RAG pipeline.")
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=Path("data/evaluation/dataset.json"),
    )
    return parser.parse_args()


def main() -> None:
    """Run real configured retrieval and generation for every dataset case."""
    args = _parse_args()
    try:
        report = evaluate_dataset(load_dataset(args.dataset), get_rag_service())
    except (OSError, ValueError, HTTPException, RagServiceUnavailableError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
