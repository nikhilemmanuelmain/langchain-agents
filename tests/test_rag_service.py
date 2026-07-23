"""Tests for grounded retrieval and answer generation."""

import logging
from collections.abc import Sequence

import pytest
from langchain_core.documents import Document

from app.generation.rag_service import (
    FALLBACK_ANSWER,
    ModelAnswer,
    RagService,
    RagServiceUnavailableError,
)


def make_chunk(
    content: str,
    *,
    chunk_id: str = "guide-chunk-0",
    document_id: str = "guide",
    filename: str = "guide.md",
    page: int | None = None,
) -> Document:
    metadata: dict[str, object] = {
        "document_id": document_id,
        "filename": filename,
        "source": f"data/documents/{filename}",
        "file_type": "markdown",
        "chunk_index": 0,
        "chunk_id": chunk_id,
    }
    if page is not None:
        metadata["page"] = page
    return Document(page_content=content, metadata=metadata)


class FakeRetriever:
    def __init__(
        self,
        documents: list[Document] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.documents = documents or []
        self.error = error
        self.calls: list[tuple[str, int, Sequence[str] | None]] = []

    def search(
        self,
        query: str,
        top_k: int = 4,
        document_ids: Sequence[str] | None = None,
    ) -> list[Document]:
        self.calls.append((query, top_k, document_ids))
        if self.error:
            raise self.error
        return self.documents


class FakeAnswerChain:
    def __init__(self, response: object = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.inputs: list[dict[str, str]] = []

    def invoke(self, input: dict[str, str]) -> object:
        self.inputs.append(input)
        if self.error:
            raise self.error
        return self.response


def test_grounded_answer_returns_only_cited_retrieved_sources() -> None:
    first = make_chunk("Reset passwords in Account Settings.")
    second = make_chunk(
        "Contact support for account recovery.",
        chunk_id="guide-chunk-1",
    )
    retriever = FakeRetriever([first, second])
    chain = FakeAnswerChain(
        ModelAnswer(
            answer="Reset it in Account Settings.",
            cited_chunk_ids=["guide-chunk-0", "unknown", "guide-chunk-0"],
        )
    )
    service = RagService(retriever=retriever, answer_chain=chain, top_k=2)

    response = service.answer("How do I reset my password?", ["guide"])

    assert response.answer == "Reset it in Account Settings."
    assert [source.chunk_id for source in response.sources] == ["guide-chunk-0"]
    assert retriever.calls == [("How do I reset my password?", 2, ["guide"])]
    assert "Reset passwords in Account Settings." in chain.inputs[0]["context"]
    assert "guide-chunk-0" in chain.inputs[0]["context"]


def test_no_retrieved_documents_returns_fallback_without_model_call() -> None:
    chain = FakeAnswerChain(error=AssertionError("must not be called"))
    service = RagService(retriever=FakeRetriever([]), answer_chain=chain)

    response = service.answer("Unknown question")

    assert response.answer == FALLBACK_ANSWER
    assert response.sources == []
    assert chain.inputs == []


def test_model_refusal_never_returns_sources() -> None:
    chain = FakeAnswerChain(
        ModelAnswer(
            answer=FALLBACK_ANSWER,
            cited_chunk_ids=["guide-chunk-0"],
        )
    )
    service = RagService(
        retriever=FakeRetriever([make_chunk("Some unrelated text")]),
        answer_chain=chain,
    )

    response = service.answer("Unsupported question")

    assert response.answer == FALLBACK_ANSWER
    assert response.sources == []


def test_answer_without_valid_citation_is_replaced_with_fallback() -> None:
    chain = FakeAnswerChain(
        {"answer": "Unsupported claim", "cited_chunk_ids": ["invented-chunk"]}
    )
    service = RagService(
        retriever=FakeRetriever([make_chunk("Documentation text")]),
        answer_chain=chain,
    )

    response = service.answer("Question")

    assert response.answer == FALLBACK_ANSWER
    assert response.sources == []


@pytest.mark.parametrize(
    "malformed_response",
    [
        None,
        {"answer": "", "cited_chunk_ids": []},
        {"answer": "Answer", "cited_chunk_ids": [], "unexpected": True},
    ],
)
def test_malformed_model_response_is_handled(malformed_response: object) -> None:
    service = RagService(
        retriever=FakeRetriever([make_chunk("Documentation text")]),
        answer_chain=FakeAnswerChain(malformed_response),
    )

    with pytest.raises(RagServiceUnavailableError, match="invalid response"):
        service.answer("Question")


def test_model_failure_is_handled() -> None:
    service = RagService(
        retriever=FakeRetriever([make_chunk("Documentation text")]),
        answer_chain=FakeAnswerChain(error=TimeoutError("provider timeout")),
    )

    with pytest.raises(RagServiceUnavailableError, match="unavailable"):
        service.answer("Question")


def test_retrieval_failure_is_handled() -> None:
    service = RagService(
        retriever=FakeRetriever(error=RuntimeError("database unavailable")),
        answer_chain=FakeAnswerChain(),
    )

    with pytest.raises(RagServiceUnavailableError, match="retrieval"):
        service.answer("Question")


def test_source_preserves_pdf_page_metadata() -> None:
    chunk = make_chunk(
        "Trial periods last six months.",
        chunk_id="handbook-page-12-chunk-0",
        document_id="handbook",
        filename="handbook.pdf",
        page=12,
    )
    service = RagService(
        retriever=FakeRetriever([chunk]),
        answer_chain=FakeAnswerChain(
            ModelAnswer(
                answer="The trial period is six months.",
                cited_chunk_ids=["handbook-page-12-chunk-0"],
            )
        ),
    )

    response = service.answer("How long is the trial period?")

    assert response.sources[0].model_dump() == {
        "document_id": "handbook",
        "filename": "handbook.pdf",
        "page": 12,
        "section": None,
        "chunk_id": "handbook-page-12-chunk-0",
    }


def test_retrieval_count_is_logged_without_question_or_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret_question = "private question text"
    secret_content = "private document content"
    service = RagService(
        retriever=FakeRetriever([make_chunk(secret_content)]),
        answer_chain=FakeAnswerChain(
            ModelAnswer(answer=FALLBACK_ANSWER, cited_chunk_ids=[])
        ),
    )

    with caplog.at_level(logging.INFO):
        service.answer(secret_question)

    assert "Retrieved 1 documentation chunks" in caplog.text
    assert secret_question not in caplog.text
    assert secret_content not in caplog.text
