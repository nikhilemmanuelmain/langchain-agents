"""Grounded documentation chat endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.conversations.service import ConversationService
from app.dependencies import get_conversation_service
from app.generation.rag_service import RagServiceUnavailableError
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    conversation_service: Annotated[
        ConversationService, Depends(get_conversation_service)
    ],
) -> ChatResponse:
    """Answer a question using only indexed documentation."""
    try:
        return conversation_service.answer(
            request.question,
            request.conversation_id,
            request.document_ids,
        )
    except RagServiceUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail="The chat service is temporarily unavailable.",
        ) from exc
