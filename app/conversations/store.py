"""Thread-safe bounded in-memory conversation history."""

from collections import OrderedDict, deque
from dataclasses import dataclass
from threading import RLock
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """One user question and assistant response."""

    user_question: str
    assistant_answer: str


class ConversationStore:
    """Keep isolated, bounded dialogue histories for one application process."""

    def __init__(self, *, max_turns: int, max_conversations: int) -> None:
        if max_turns <= 0 or max_conversations <= 0:
            raise ValueError("Conversation limits must be greater than zero.")
        self._max_turns = max_turns
        self._max_conversations = max_conversations
        self._conversations: OrderedDict[str, deque[ConversationTurn]] = OrderedDict()
        self._lock = RLock()

    def resolve_id(self, conversation_id: str | None) -> str:
        """Return the supplied ID or create a new unpredictable UUID."""
        resolved = conversation_id or str(uuid4())
        with self._lock:
            if resolved not in self._conversations:
                self._conversations[resolved] = deque(maxlen=self._max_turns)
                self._evict_excess_conversations()
            else:
                self._conversations.move_to_end(resolved)
        return resolved

    def get_history(self, conversation_id: str) -> list[ConversationTurn]:
        """Return a copy of one conversation without exposing mutable state."""
        with self._lock:
            history = self._conversations.get(conversation_id)
            if history is None:
                return []
            self._conversations.move_to_end(conversation_id)
            return list(history)

    def append(self, conversation_id: str, turn: ConversationTurn) -> None:
        """Append one turn while enforcing per-conversation limits."""
        with self._lock:
            history = self._conversations.setdefault(
                conversation_id, deque(maxlen=self._max_turns)
            )
            history.append(turn)
            self._conversations.move_to_end(conversation_id)
            self._evict_excess_conversations()

    def _evict_excess_conversations(self) -> None:
        while len(self._conversations) > self._max_conversations:
            self._conversations.popitem(last=False)
