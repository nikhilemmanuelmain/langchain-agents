"""Tests for the temporary chat endpoint."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_chat_returns_temporary_response() -> None:
    response = client.post(
        "/api/chat", json={"question": "How can I reset my password?"}
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "The chat API is connected.",
        "sources": [],
    }


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_chat_rejects_empty_question(question: str) -> None:
    response = client.post("/api/chat", json={"question": question})

    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


def test_chat_requires_question() -> None:
    response = client.post("/api/chat", json={})

    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"
