"""OpenAI chat-model and structured-output construction."""

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from app.config import Settings
from app.generation.prompts import ANSWER_PROMPT
from app.generation.rag_service import ModelAnswer


class ChatModelConfigurationError(ValueError):
    """Raised when the configured chat model cannot be constructed."""


def create_chat_model(settings: Settings) -> BaseChatModel:
    """Create the replaceable OpenAI chat-model implementation."""
    if (
        settings.openai_api_key is None
        or not settings.openai_api_key.get_secret_value()
    ):
        raise ChatModelConfigurationError(
            "OPENAI_API_KEY is required to create the chat model."
        )
    if not settings.openai_chat_model.strip():
        raise ChatModelConfigurationError("OPENAI_CHAT_MODEL must not be empty.")
    return ChatOpenAI(
        model=settings.openai_chat_model,
        api_key=settings.openai_api_key,
        timeout=settings.openai_request_timeout_seconds,
        max_retries=settings.openai_max_retries,
        use_responses_api=True,
    )


def create_answer_chain(
    chat_model: BaseChatModel,
) -> Runnable[dict[str, str], ModelAnswer]:
    """Create a prompt-to-strict-structured-answer chain."""
    structured_model = chat_model.with_structured_output(
        ModelAnswer,
        method="json_schema",
        strict=True,
    )
    return ANSWER_PROMPT | structured_model
