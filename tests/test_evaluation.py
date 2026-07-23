"""Tests for separate RAG evaluation metrics and dataset validation."""

from pathlib import Path

from app.evaluation.evaluator import evaluate_dataset, load_dataset
from app.evaluation.models import EvaluationCase
from app.generation.rag_service import FALLBACK_ANSWER
from app.schemas.chat import ChatResponse, SourceReference


class FakeAnswerer:
    def __init__(self, responses: dict[str, ChatResponse]) -> None:
        self.responses = responses

    def answer(self, question: str) -> ChatResponse:
        return self.responses[question]


def test_evaluation_reports_metrics_separately() -> None:
    answerable = EvaluationCase(
        case_id="trial",
        question="How long is the trial period?",
        expected_source="employee-guide.pdf",
        expected_answer_terms=["six months"],
        answerable=True,
    )
    unsupported = EvaluationCase(
        case_id="travel",
        question="What is the travel rate?",
        answerable=False,
    )
    answerer = FakeAnswerer(
        {
            answerable.question: ChatResponse(
                answer="The trial period is six months.",
                sources=[
                    SourceReference(
                        document_id="employee-guide",
                        filename="employee-guide.pdf",
                        page=3,
                        chunk_id="employee-guide-page-3-chunk-0",
                    )
                ],
            ),
            unsupported.question: ChatResponse(answer=FALLBACK_ANSWER, sources=[]),
        }
    )

    report = evaluate_dataset([answerable, unsupported], answerer)

    assert report.summary.case_count == 2
    assert report.summary.retrieval_accuracy == 1
    assert report.summary.answer_correctness == 1
    assert report.summary.groundedness == 1
    assert report.summary.citation_correctness == 1
    assert report.summary.unsupported_question_refusal == 1


def test_bad_answer_can_fail_metrics_independently() -> None:
    case = EvaluationCase(
        case_id="trial",
        question="How long is the trial period?",
        expected_source="employee-guide.pdf",
        expected_answer_terms=["six months"],
        answerable=True,
    )
    answerer = FakeAnswerer(
        {
            case.question: ChatResponse(
                answer="It is three months.",
                sources=[
                    SourceReference(
                        document_id="employee-guide",
                        filename="employee-guide.pdf",
                        chunk_id="wrong-chunk",
                    )
                ],
            )
        }
    )

    result = evaluate_dataset([case], answerer).results[0]

    assert result.metrics.retrieval_accuracy is True
    assert result.metrics.answer_correctness is False
    assert result.metrics.groundedness is True
    assert result.metrics.citation_correctness is True


def test_checked_in_dataset_is_valid() -> None:
    dataset = Path("data/evaluation/dataset.json")

    cases = load_dataset(dataset)

    assert len(cases) == 3
    assert any(case.answerable for case in cases)
    assert any(not case.answerable for case in cases)
