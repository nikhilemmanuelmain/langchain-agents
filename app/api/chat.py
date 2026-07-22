"""Temporary chat endpoint for the backend foundation."""

from fastapi import APIRouter

from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Confirm connectivity until retrieval is implemented in later modules."""
    del request
    return ChatResponse(answer="The chat API is connected.", sources=[])
