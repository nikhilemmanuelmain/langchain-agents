"""Chat request and response schemas."""

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """A question submitted to the chat API."""

    question: str = Field(
        max_length=2000,
        description="The documentation question to answer (maximum 2000 characters).",
    )
    document_ids: list[str] | None = Field(
        default=None,
        description="Optional document IDs to restrict retrieval.",
    )
    conversation_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="Existing conversation ID; omit to start a conversation.",
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_empty(cls, value: str) -> str:
        """Reject blank questions and normalize surrounding whitespace."""
        question = value.strip()
        if not question:
            raise ValueError("Question must not be empty.")
        return question

    @field_validator("document_ids")
    @classmethod
    def document_ids_must_not_be_empty(
        cls, value: list[str] | None
    ) -> list[str] | None:
        """Normalize document filters and reject empty identifiers."""
        if value is None:
            return None
        normalized = [document_id.strip() for document_id in value]
        if any(not document_id for document_id in normalized):
            raise ValueError("Document IDs must not contain empty values.")
        return list(dict.fromkeys(normalized))

    @field_validator("conversation_id")
    @classmethod
    def normalize_conversation_id(cls, value: str | None) -> str | None:
        """Normalize an optional conversation identifier."""
        return value.strip() if value is not None else None


class SourceReference(BaseModel):
    """A retrieved chunk that directly supports the generated answer."""

    document_id: str
    filename: str
    page: int | None = None
    section: str | None = None
    chunk_id: str


class ChatResponse(BaseModel):
    """A grounded documentation answer and its supporting sources."""

    answer: str
    sources: list[SourceReference]
    conversation_id: str | None = None
