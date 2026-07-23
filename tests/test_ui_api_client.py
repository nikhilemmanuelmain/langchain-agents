"""Focused contract and failure tests for the Streamlit HTTP client."""

import httpx
import pytest

from test_ui.api_client import (
    ApiRequestError,
    ApiTimeoutError,
    ApiUnavailableError,
    ChatbotApiClient,
    InvalidApiResponseError,
)


def _client(handler: httpx.MockTransport) -> ChatbotApiClient:
    return ChatbotApiClient("http://backend.test", transport=handler)


def test_document_operations_use_existing_api_contract() -> None:
    requests: list[httpx.Request] = []
    document = {
        "document_id": "guide-123",
        "filename": "guide.md",
        "checksum": "abc",
        "status": "indexed",
        "chunk_count": 2,
        "indexed_at": "2026-01-01T00:00:00Z",
        "error": None,
    }

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json=[document])
        if request.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(200, json=document)

    client = _client(httpx.MockTransport(handle))
    try:
        assert client.list_documents() == [document]
        assert client.upload_document("guide.md", b"content") == document
        assert client.reindex_document("guide-123") == document
        client.delete_document("guide-123")
    finally:
        client.close()

    assert [(request.method, request.url.path) for request in requests] == [
        ("GET", "/api/documents"),
        ("POST", "/api/documents"),
        ("POST", "/api/documents/guide-123/reindex"),
        ("DELETE", "/api/documents/guide-123"),
    ]
    assert b'name="file"; filename="guide.md"' in requests[1].content


def test_chat_sends_conversation_and_document_filters() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/chat"
        assert request.read() == (
            b'{"question":"Follow up","conversation_id":"conversation-1",'
            b'"document_ids":["guide-123"]}'
        )
        return httpx.Response(
            200,
            json={
                "answer": "Grounded answer.",
                "sources": [
                    {
                        "document_id": "guide-123",
                        "filename": "guide.md",
                        "page": None,
                        "section": "Passwords",
                        "chunk_id": "chunk-1",
                    }
                ],
                "conversation_id": "conversation-1",
            },
        )

    client = _client(httpx.MockTransport(handle))
    try:
        response = client.chat(
            "Follow up",
            conversation_id="conversation-1",
            document_ids=["guide-123"],
        )
    finally:
        client.close()

    assert response["answer"] == "Grounded answer."
    assert response["conversation_id"] == "conversation-1"


def test_api_error_uses_safe_backend_detail() -> None:
    client = _client(
        httpx.MockTransport(
            lambda request: httpx.Response(
                415,
                json={
                    "error": "http_error",
                    "detail": "Unsupported document type '.csv'.",
                },
            )
        )
    )

    with pytest.raises(ApiRequestError, match="Unsupported document type") as exc_info:
        client.upload_document("data.csv", b"a,b")
    client.close()

    assert exc_info.value.status_code == 415


def test_invalid_json_response_is_rejected() -> None:
    client = _client(
        httpx.MockTransport(
            lambda request: httpx.Response(200, text="<html>not json</html>")
        )
    )

    with pytest.raises(InvalidApiResponseError, match="not valid JSON"):
        client.list_documents()
    client.close()


@pytest.mark.parametrize(
    ("exception", "expected_error"),
    [
        (httpx.ConnectError("offline"), ApiUnavailableError),
        (httpx.ReadTimeout("slow"), ApiTimeoutError),
    ],
)
def test_transport_failures_have_helpful_errors(
    exception: httpx.RequestError,
    expected_error: type[Exception],
) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        raise exception

    client = _client(httpx.MockTransport(handle))
    with pytest.raises(expected_error):
        client.list_documents()
    client.close()


def test_invalid_chat_shape_is_rejected() -> None:
    client = _client(
        httpx.MockTransport(
            lambda request: httpx.Response(200, json={"answer": "Missing sources"})
        )
    )

    with pytest.raises(InvalidApiResponseError, match="invalid chat response"):
        client.chat("Question")
    client.close()
