"""Tests for request IDs and structured privacy-conscious logs."""

import json
import logging

from fastapi.testclient import TestClient

from app.main import app
from app.observability import JsonLogFormatter

client = TestClient(app)


def test_request_id_is_preserved_for_safe_client_value() -> None:
    response = client.get("/health", headers={"X-Request-ID": "request-123"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "request-123"


def test_invalid_request_id_is_replaced() -> None:
    response = client.get("/health", headers={"X-Request-ID": "bad id!"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] != "bad id!"
    assert response.headers["X-Request-ID"]


def test_json_log_formatter_emits_structured_metadata_only() -> None:
    record = logging.LogRecord(
        name="app.requests",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="request_completed",
        args=(),
        exc_info=None,
    )
    record.request_id = "request-123"
    record.method = "POST"
    record.path = "/api/chat"
    record.status_code = 200
    record.duration_ms = 12.5

    payload = json.loads(JsonLogFormatter().format(record))

    assert payload["request_id"] == "request-123"
    assert payload["method"] == "POST"
    assert payload["path"] == "/api/chat"
    assert "question" not in payload
    assert "content" not in payload
