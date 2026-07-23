"""Run deterministic, separately reported RAG quality checks."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from pydantic import TypeAdapter

from app.evaluation.models import (
    EvaluationCase,
    EvaluationMetrics,
    EvaluationReport,
    EvaluationResult,
    EvaluationSummary,
)
from app.generation.rag_service import FALLBACK_ANSWER
from app.schemas.chat import ChatResponse


class EvaluationAnswerer(Protocol):
    def answer(self, question: str) -> ChatResponse: ...


def load_dataset(path: str | Path) -> list[EvaluationCase]:
    """Load and validate an evaluation dataset from JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return TypeAdapter(list[EvaluationCase]).validate_python(payload)


def evaluate_dataset(
    cases: Sequence[EvaluationCase], answerer: EvaluationAnswerer
) -> EvaluationReport:
    """Evaluate retrieval, answer, grounding, citations, and refusal separately."""
    results = [_evaluate_case(case, answerer.answer(case.question)) for case in cases]
    return EvaluationReport(results=results, summary=_summarize(results))


def _evaluate_case(case: EvaluationCase, response: ChatResponse) -> EvaluationResult:
    filenames = list(dict.fromkeys(source.filename for source in response.sources))
    answer_lower = response.answer.casefold()
    contains_terms = all(
        term.casefold() in answer_lower for term in case.expected_answer_terms
    )
    refused = response.answer.strip() == FALLBACK_ANSWER and not response.sources

    if case.answerable:
        retrieval_accuracy = case.expected_source in filenames
        answer_correctness = contains_terms and not refused
        groundedness = bool(response.sources) and not refused
        citation_correctness = case.expected_source in filenames and all(
            source.chunk_id for source in response.sources
        )
        unsupported_refusal = True
    else:
        retrieval_accuracy = not response.sources
        answer_correctness = refused
        groundedness = refused
        citation_correctness = not response.sources
        unsupported_refusal = refused

    return EvaluationResult(
        case_id=case.case_id,
        question=case.question,
        answer=response.answer,
        source_filenames=filenames,
        metrics=EvaluationMetrics(
            retrieval_accuracy=retrieval_accuracy,
            answer_correctness=answer_correctness,
            groundedness=groundedness,
            citation_correctness=citation_correctness,
            unsupported_question_refusal=unsupported_refusal,
        ),
    )


def _summarize(results: Sequence[EvaluationResult]) -> EvaluationSummary:
    if not results:
        return EvaluationSummary(
            case_count=0,
            retrieval_accuracy=0,
            answer_correctness=0,
            groundedness=0,
            citation_correctness=0,
            unsupported_question_refusal=0,
        )

    def rate(metric: str) -> float:
        passed = sum(bool(getattr(result.metrics, metric)) for result in results)
        return passed / len(results)

    return EvaluationSummary(
        case_count=len(results),
        retrieval_accuracy=rate("retrieval_accuracy"),
        answer_correctness=rate("answer_correctness"),
        groundedness=rate("groundedness"),
        citation_correctness=rate("citation_correctness"),
        unsupported_question_refusal=rate("unsupported_question_refusal"),
    )
