"""Centralized API exception handlers."""

import logging

from fastapi import HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


async def validation_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Return validation failures in a consistent JSON envelope."""
    del request
    if not isinstance(exc, RequestValidationError):
        raise TypeError("Expected RequestValidationError.")
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "detail": jsonable_encoder(exc.errors()),
        },
    )


async def http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return explicitly raised HTTP errors in a consistent JSON envelope."""
    del request
    if not isinstance(exc, HTTPException):
        raise TypeError("Expected HTTPException.")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "http_error", "detail": exc.detail},
        headers=exc.headers,
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Log unexpected failures and return a safe public response."""
    logger.exception(
        "Unhandled error while processing %s %s",
        request.method,
        request.url.path,
        exc_info=exc,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "detail": "An unexpected error occurred.",
        },
    )
