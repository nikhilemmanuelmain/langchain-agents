"""Chat request and response schemas."""

from typing import Any

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """A question submitted to the chat API."""

    question: str = Field(description="The documentation question to answer.")

    @field_validator("question")
    @classmethod
    def question_must_not_be_empty(cls, value: str) -> str:
        """Reject blank questions and normalize surrounding whitespace."""
        question = value.strip()
        if not question:
            raise ValueError("Question must not be empty.")
        return question


class ChatResponse(BaseModel):
    """The temporary chat response contract."""

    answer: str
    sources: list[dict[str, Any]]
