"""Tests for application configuration and CORS."""

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def test_cors_uses_configured_origins() -> None:
    settings = Settings(cors_allowed_origins="https://docs.example.com")
    client = TestClient(create_app(settings))

    response = client.options(
        "/api/chat",
        headers={
            "Origin": "https://docs.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == (
        "https://docs.example.com"
    )


def test_cors_origin_parser_ignores_blanks() -> None:
    settings = Settings(
        cors_allowed_origins=" https://one.example, ,https://two.example "
    )

    assert settings.cors_origins == [
        "https://one.example",
        "https://two.example",
    ]
