"""Request IDs and privacy-conscious structured request logging."""

import json
import logging
import re
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from datetime import UTC, datetime
from time import perf_counter
from uuid import uuid4

from fastapi import Request, Response

REQUEST_ID_HEADER = "X-Request-ID"
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
request_id_context: ContextVar[str] = ContextVar("request_id", default="-")
logger = logging.getLogger("app.requests")


class JsonLogFormatter(logging.Formatter):
    """Render application logs as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", request_id_context.get()),
        }
        for key in ("method", "path", "status_code", "duration_ms"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload, separators=(",", ":"))


def configure_logging(level: str) -> None:
    """Configure the application's logger without replacing host loggers."""
    application_logger = logging.getLogger("app")
    if not application_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonLogFormatter())
        application_logger.addHandler(handler)
    application_logger.setLevel(level)
    application_logger.propagate = False


async def request_observability_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Attach a safe request ID and log request metadata without bodies."""
    incoming_id = request.headers.get(REQUEST_ID_HEADER, "")
    request_id = (
        incoming_id if REQUEST_ID_PATTERN.fullmatch(incoming_id) else str(uuid4())
    )
    token = request_id_context.set(request_id)
    started = perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers[REQUEST_ID_HEADER] = request_id
        return response
    finally:
        duration_ms = round((perf_counter() - started) * 1000, 2)
        logger.info(
            "request_completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "duration_ms": duration_ms,
            },
        )
        request_id_context.reset(token)
