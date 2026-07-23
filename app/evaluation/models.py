"""Schemas for reproducible RAG evaluation datasets and reports."""

from pydantic import BaseModel, Field, model_validator


class EvaluationCase(BaseModel):
    """Expected behavior for one documentation question."""

    case_id: str
    question: str
    expected_source: str | None = None
    expected_answer_terms: list[str] = Field(default_factory=list)
    answerable: bool

    @model_validator(mode="after")
    def validate_answerable_expectations(self) -> "EvaluationCase":
        if self.answerable and not self.expected_source:
            raise ValueError("Answerable cases require expected_source.")
        return self


class EvaluationMetrics(BaseModel):
    retrieval_accuracy: bool
    answer_correctness: bool
    groundedness: bool
    citation_correctness: bool
    unsupported_question_refusal: bool


class EvaluationResult(BaseModel):
    case_id: str
    question: str
    answer: str
    source_filenames: list[str]
    metrics: EvaluationMetrics


class EvaluationSummary(BaseModel):
    case_count: int
    retrieval_accuracy: float
    answer_correctness: float
    groundedness: float
    citation_correctness: float
    unsupported_question_refusal: float


class EvaluationReport(BaseModel):
    results: list[EvaluationResult]
    summary: EvaluationSummary
