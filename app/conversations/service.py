"""Conversation-aware query rewriting and grounded answer coordination."""

import logging
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.conversations.store import ConversationStore, ConversationTurn
from app.generation.rag_service import RagService, RagServiceUnavailableError
from app.schemas.chat import ChatResponse

logger = logging.getLogger(__name__)


class StandaloneQuery(BaseModel):
    """Strict result from follow-up query rewriting."""

    model_config = ConfigDict(extra="forbid")

    standalone_question: str = Field(min_length=1)


class RewriteChain(Protocol):
    def invoke(self, input: dict[str, str]) -> object: ...


class ConversationService:
    """Rewrite follow-ups, call grounded RAG, and retain bounded dialogue."""

    def __init__(
        self,
        *,
        rag_service: RagService,
        rewrite_chain: RewriteChain,
        store: ConversationStore,
    ) -> None:
        self._rag_service = rag_service
        self._rewrite_chain = rewrite_chain
        self._store = store

    def answer(
        self,
        question: str,
        conversation_id: str | None = None,
        document_ids: Sequence[str] | None = None,
    ) -> ChatResponse:
        """Answer a standalone or context-dependent documentation question."""
        resolved_id = self._store.resolve_id(conversation_id)
        history = self._store.get_history(resolved_id)
        retrieval_query = question
        if history:
            try:
                raw_rewrite = self._rewrite_chain.invoke(
                    {
                        "history": _format_history(history),
                        "question": question,
                    }
                )
                retrieval_query = StandaloneQuery.model_validate(
                    raw_rewrite
                ).standalone_question.strip()
            except (ValidationError, TypeError, ValueError) as exc:
                logger.error(
                    "Follow-up rewrite returned invalid output; error_type=%s",
                    type(exc).__name__,
                )
                raise RagServiceUnavailableError(
                    "Follow-up question rewriting returned an invalid response."
                ) from exc
            except Exception as exc:
                logger.error(
                    "Follow-up question rewriting failed; error_type=%s",
                    type(exc).__name__,
                )
                raise RagServiceUnavailableError(
                    "Follow-up question rewriting is currently unavailable."
                ) from exc

        response = self._rag_service.answer(
            question,
            document_ids,
            retrieval_query=retrieval_query,
        )
        self._store.append(
            resolved_id,
            ConversationTurn(
                user_question=question,
                assistant_answer=response.answer,
            ),
        )
        return response.model_copy(update={"conversation_id": resolved_id})


def _format_history(history: Sequence[ConversationTurn]) -> str:
    """Format dialogue for reference resolution, not documentary evidence."""
    return "\n".join(
        f"User: {turn.user_question}\nAssistant: {turn.assistant_answer}"
        for turn in history
    )
