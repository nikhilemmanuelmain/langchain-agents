"""Deterministic retrieval-augmented answer orchestration."""

import json
import logging
from collections.abc import Sequence
from typing import Any, Protocol

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.schemas.chat import ChatResponse, SourceReference

logger = logging.getLogger(__name__)

FALLBACK_ANSWER = "I could not find this information in the available documentation."


class ModelAnswer(BaseModel):
    """Strict output requested from the chat model."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(
        min_length=1,
        description="A concise answer grounded only in the supplied chunks.",
    )
    cited_chunk_ids: list[str] = Field(
        description="Chunk IDs from the supplied context that support the answer."
    )


class Retriever(Protocol):
    """Replacement boundary for vector retrieval."""

    def search(
        self,
        query: str,
        top_k: int = 4,
        document_ids: Sequence[str] | None = None,
    ) -> list[Document]: ...


class AnswerChain(Protocol):
    """Replacement boundary for structured answer generation."""

    def invoke(self, input: dict[str, str]) -> object: ...


class RagServiceUnavailableError(RuntimeError):
    """Raised when retrieval or generation cannot complete safely."""


class RagService:
    """Retrieve context, generate an answer, and verify its citations."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        answer_chain: AnswerChain,
        top_k: int = 4,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        self._retriever = retriever
        self._answer_chain = answer_chain
        self._top_k = top_k

    def answer(
        self,
        question: str,
        document_ids: Sequence[str] | None = None,
        *,
        retrieval_query: str | None = None,
    ) -> ChatResponse:
        """Return an answer grounded only in retrieved documentation chunks."""
        search_query = retrieval_query or question
        try:
            documents = self._retriever.search(
                search_query,
                top_k=self._top_k,
                document_ids=document_ids,
            )
        except Exception as exc:
            logger.error(
                "Documentation retrieval failed; error_type=%s",
                type(exc).__name__,
            )
            raise RagServiceUnavailableError(
                "Documentation retrieval is currently unavailable."
            ) from exc

        logger.info("Retrieved %d documentation chunks", len(documents))
        if not documents:
            return _fallback_response()

        try:
            chunks_by_id = _map_chunks_by_id(documents)
            context = _format_context(documents)
            raw_response = self._answer_chain.invoke(
                {"question": question, "context": context}
            )
            model_answer = ModelAnswer.model_validate(raw_response)
        except (ValidationError, TypeError, ValueError) as exc:
            logger.error(
                "Chat model returned an invalid grounded-answer response; "
                "error_type=%s",
                type(exc).__name__,
            )
            raise RagServiceUnavailableError(
                "Answer generation returned an invalid response."
            ) from exc
        except Exception as exc:
            logger.error(
                "Grounded answer generation failed; error_type=%s",
                type(exc).__name__,
            )
            raise RagServiceUnavailableError(
                "Answer generation is currently unavailable."
            ) from exc

        answer = model_answer.answer.strip()
        if answer == FALLBACK_ANSWER:
            return _fallback_response()

        cited_ids = list(dict.fromkeys(model_answer.cited_chunk_ids))
        valid_cited_ids = [
            chunk_id for chunk_id in cited_ids if chunk_id in chunks_by_id
        ]
        if not valid_cited_ids:
            logger.warning("Generated answer had no valid retrieved-chunk citations")
            return _fallback_response()

        try:
            sources = [
                _source_from_document(chunks_by_id[chunk_id])
                for chunk_id in valid_cited_ids
            ]
        except (ValidationError, TypeError, ValueError) as exc:
            logger.error(
                "Retrieved chunk contained invalid source metadata; error_type=%s",
                type(exc).__name__,
            )
            raise RagServiceUnavailableError(
                "Retrieved documentation metadata is invalid."
            ) from exc
        return ChatResponse(answer=answer, sources=sources)


def _fallback_response() -> ChatResponse:
    return ChatResponse(answer=FALLBACK_ANSWER, sources=[])


def _chunk_id(document: Document) -> str:
    value = document.metadata.get("chunk_id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Retrieved chunks must have a non-empty chunk_id.")
    return value


def _map_chunks_by_id(documents: Sequence[Document]) -> dict[str, Document]:
    return {_chunk_id(document): document for document in documents}


def _format_context(documents: Sequence[Document]) -> str:
    sections = []
    for document in documents:
        public_metadata = {
            key: document.metadata.get(key)
            for key in ("chunk_id", "document_id", "filename", "page", "section")
            if document.metadata.get(key) is not None
        }
        sections.append(
            "CHUNK_METADATA: "
            f"{json.dumps(public_metadata, ensure_ascii=False, sort_keys=True)}\n"
            f"CHUNK_CONTENT:\n{document.page_content}"
        )
    return "\n\n---\n\n".join(sections)


def _source_from_document(document: Document) -> SourceReference:
    metadata: dict[str, Any] = document.metadata
    return SourceReference(
        document_id=_required_source_metadata(metadata, "document_id"),
        filename=_required_source_metadata(metadata, "filename"),
        page=metadata.get("page"),
        section=metadata.get("section"),
        chunk_id=_chunk_id(document),
    )


def _required_source_metadata(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Retrieved chunks must have a non-empty {key}.")
    return value
