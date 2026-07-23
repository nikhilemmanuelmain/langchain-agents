"""Tests for the grounded chat endpoint."""

from collections.abc import Sequence

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_conversation_service
from app.generation.rag_service import FALLBACK_ANSWER, RagServiceUnavailableError
from app.main import app
from app.schemas.chat import ChatResponse, SourceReference

client = TestClient(app)


class FakeConversationService:
    """Predictable route dependency that records request filters."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Sequence[str] | None]] = []

    def answer(
        self,
        question: str,
        conversation_id: str | None = None,
        document_ids: Sequence[str] | None = None,
    ) -> ChatResponse:
        resolved_id = conversation_id or "generated-conversation-id"
        self.calls.append((question, document_ids))
        return ChatResponse(
            answer="Use the Account Settings page.",
            sources=[
                SourceReference(
                    document_id="guide",
                    filename="guide.md",
                    chunk_id="guide-chunk-0",
                )
            ],
            conversation_id=resolved_id,
        )


@pytest.fixture(autouse=True)
def override_rag_service() -> FakeConversationService:
    service = FakeConversationService()
    app.dependency_overrides[get_conversation_service] = lambda: service
    yield service
    app.dependency_overrides.clear()


def test_chat_returns_grounded_response(
    override_rag_service: FakeConversationService,
) -> None:
    response = client.post(
        "/api/chat",
        json={
            "question": "How can I reset my password?",
            "document_ids": ["guide", "guide"],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Use the Account Settings page.",
        "conversation_id": "generated-conversation-id",
        "sources": [
            {
                "document_id": "guide",
                "filename": "guide.md",
                "page": None,
                "section": None,
                "chunk_id": "guide-chunk-0",
            }
        ],
    }
    assert override_rag_service.calls == [("How can I reset my password?", ["guide"])]


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_chat_rejects_empty_question(question: str) -> None:
    response = client.post("/api/chat", json={"question": question})

    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


def test_chat_requires_question() -> None:
    response = client.post("/api/chat", json={})

    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


def test_chat_rejects_empty_document_id() -> None:
    response = client.post(
        "/api/chat",
        json={"question": "Question", "document_ids": ["guide", " "]},
    )

    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


def test_chat_returns_503_when_rag_service_fails() -> None:
    class UnavailableService:
        def answer(
            self,
            question: str,
            conversation_id: str | None = None,
            document_ids: Sequence[str] | None = None,
        ) -> ChatResponse:
            del question, conversation_id, document_ids
            raise RagServiceUnavailableError("provider failed")

    app.dependency_overrides[get_conversation_service] = UnavailableService

    response = client.post("/api/chat", json={"question": "Question"})

    assert response.status_code == 503
    assert response.json() == {
        "error": "http_error",
        "detail": "The chat service is temporarily unavailable.",
    }


def test_fallback_answer_contract() -> None:
    response = ChatResponse(answer=FALLBACK_ANSWER, sources=[])

    assert response.sources == []
