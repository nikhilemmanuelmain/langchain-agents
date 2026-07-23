"""FastAPI application entry point."""

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.config import Settings, get_settings
from app.errors import (
    http_exception_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)
from app.observability import configure_logging, request_observability_middleware


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    application_settings = settings or get_settings()
    configure_logging(application_settings.log_level)
    application = FastAPI(title=application_settings.app_name)
    application.middleware("http")(request_observability_middleware)

    application.add_middleware(
        CORSMiddleware,
        allow_origins=application_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.add_exception_handler(
        RequestValidationError, validation_exception_handler
    )
    application.add_exception_handler(HTTPException, http_exception_handler)
    application.add_exception_handler(Exception, unhandled_exception_handler)

    application.include_router(health_router)
    application.include_router(chat_router)
    application.include_router(documents_router)
    return application


app = create_app()
