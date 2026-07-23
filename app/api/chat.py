"""Grounded documentation chat endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_rag_service
from app.generation.rag_service import RagService, RagServiceUnavailableError
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    rag_service: Annotated[RagService, Depends(get_rag_service)],
) -> ChatResponse:
    """Answer a question using only indexed documentation."""
    try:
        return rag_service.answer(request.question, request.document_ids)
    except RagServiceUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail="The chat service is temporarily unavailable.",
        ) from exc
