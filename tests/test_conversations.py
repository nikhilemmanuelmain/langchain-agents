"""Tests for isolated bounded conversation follow-ups."""

from collections.abc import Sequence

import pytest

from app.conversations.service import ConversationService, StandaloneQuery
from app.conversations.store import ConversationStore, ConversationTurn
from app.generation.rag_service import RagServiceUnavailableError
from app.schemas.chat import ChatResponse


class FakeRagService:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, Sequence[str] | None, str | None]] = []

    def answer(
        self,
        question: str,
        document_ids: Sequence[str] | None = None,
        *,
        retrieval_query: str | None = None,
    ) -> ChatResponse:
        self.calls.append((question, document_ids, retrieval_query))
        if self.fail:
            raise RagServiceUnavailableError("RAG failed")
        return ChatResponse(answer=f"Answer to: {question}", sources=[])


class FakeRewriteChain:
    def __init__(self, response: object | None = None) -> None:
        self.response = response or StandaloneQuery(
            standalone_question="Can annual leave carry into next year?"
        )
        self.inputs: list[dict[str, str]] = []

    def invoke(self, input: dict[str, str]) -> object:
        self.inputs.append(input)
        return self.response


def make_service(
    *,
    max_turns: int = 6,
    rag_service: FakeRagService | None = None,
    rewrite_chain: FakeRewriteChain | None = None,
) -> tuple[ConversationService, FakeRagService, FakeRewriteChain, ConversationStore]:
    rag = rag_service or FakeRagService()
    rewrite = rewrite_chain or FakeRewriteChain()
    store = ConversationStore(max_turns=max_turns, max_conversations=10)
    service = ConversationService(
        rag_service=rag,  # type: ignore[arg-type]
        rewrite_chain=rewrite,
        store=store,
    )
    return service, rag, rewrite, store


def test_first_question_creates_conversation_without_rewrite() -> None:
    service, rag, rewrite, _ = make_service()

    response = service.answer("How many annual leave days are provided?")

    assert response.conversation_id
    assert rewrite.inputs == []
    assert rag.calls == [
        (
            "How many annual leave days are provided?",
            None,
            "How many annual leave days are provided?",
        )
    ]


def test_follow_up_is_rewritten_for_retrieval_only() -> None:
    service, rag, rewrite, _ = make_service()
    first = service.answer("How many annual leave days are provided?")

    second = service.answer(
        "Can they carry them into next year?", first.conversation_id
    )

    assert second.conversation_id == first.conversation_id
    assert rag.calls[-1] == (
        "Can they carry them into next year?",
        None,
        "Can annual leave carry into next year?",
    )
    assert "How many annual leave days" in rewrite.inputs[0]["history"]
    assert "Answer to:" in rewrite.inputs[0]["history"]


def test_conversations_are_isolated() -> None:
    service, _, rewrite, _ = make_service()
    first_a = service.answer("Question for A")
    first_b = service.answer("Question for B")

    service.answer("Follow-up A", first_a.conversation_id)
    history_a = rewrite.inputs[-1]["history"]
    service.answer("Follow-up B", first_b.conversation_id)
    history_b = rewrite.inputs[-1]["history"]

    assert "Question for A" in history_a
    assert "Question for B" not in history_a
    assert "Question for B" in history_b
    assert "Question for A" not in history_b


def test_history_is_limited_to_configured_turns() -> None:
    store = ConversationStore(max_turns=2, max_conversations=10)
    conversation_id = store.resolve_id("conversation")
    for index in range(3):
        store.append(
            conversation_id,
            ConversationTurn(f"question-{index}", f"answer-{index}"),
        )

    history = store.get_history(conversation_id)

    assert [turn.user_question for turn in history] == ["question-1", "question-2"]


def test_oldest_conversation_is_evicted() -> None:
    store = ConversationStore(max_turns=2, max_conversations=2)
    first = store.resolve_id("first")
    store.append(first, ConversationTurn("first question", "first answer"))
    store.resolve_id("second")
    store.resolve_id("third")

    assert store.get_history("first") == []


def test_invalid_rewrite_is_handled_without_answering() -> None:
    rewrite = FakeRewriteChain(response={"standalone_question": "", "extra": True})
    service, rag, _, _ = make_service(rewrite_chain=rewrite)
    first = service.answer("First question")

    with pytest.raises(RagServiceUnavailableError, match="invalid response"):
        service.answer("Follow-up", first.conversation_id)

    assert len(rag.calls) == 1


def test_failed_answer_is_not_added_to_history() -> None:
    rag = FakeRagService(fail=True)
    service, _, _, store = make_service(rag_service=rag)

    with pytest.raises(RagServiceUnavailableError):
        service.answer("Question", "conversation")

    assert store.get_history("conversation") == []
