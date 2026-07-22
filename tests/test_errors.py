"""Tests for centralized error responses."""

import asyncio

from fastapi import Request

from app.errors import unhandled_exception_handler


def test_unhandled_error_response_hides_exception_details() -> None:
    request = Request(
        {"type": "http", "method": "GET", "path": "/failure", "headers": []}
    )

    response = asyncio.run(
        unhandled_exception_handler(request, RuntimeError("private detail"))
    )

    assert response.status_code == 500
    assert response.body == (
        b'{"error":"internal_server_error","detail":"An unexpected error occurred."}'
    )
