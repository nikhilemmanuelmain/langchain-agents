"""HTTP client for the documentation chatbot's public FastAPI contract."""

from collections.abc import Mapping, Sequence
from typing import Any

import httpx

DEFAULT_CONNECT_TIMEOUT_SECONDS = 5.0
DEFAULT_RESPONSE_TIMEOUT_SECONDS = 60.0


class ChatbotApiError(RuntimeError):
    """Base error shown safely by the manual test UI."""


class ApiUnavailableError(ChatbotApiError):
    """Raised when the FastAPI service cannot be reached."""


class ApiTimeoutError(ChatbotApiError):
    """Raised when an API request exceeds its configured timeout."""


class InvalidApiResponseError(ChatbotApiError):
    """Raised when the API response does not match its public contract."""


class ApiRequestError(ChatbotApiError):
    """Raised when FastAPI returns a non-success status."""

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class ChatbotApiClient:
    """Small synchronous client used only by the Streamlit test application."""

    def __init__(
        self,
        base_url: str,
        *,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
        response_timeout: float = DEFAULT_RESPONSE_TIMEOUT_SECONDS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        normalized_url = base_url.strip().rstrip("/")
        if not normalized_url:
            raise ValueError("The chatbot API URL must not be empty.")
        self._client = httpx.Client(
            base_url=normalized_url,
            timeout=httpx.Timeout(response_timeout, connect=connect_timeout),
            transport=transport,
        )

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()

    def list_documents(self) -> list[dict[str, Any]]:
        """Return public metadata for all managed documents."""
        payload = self._request_json("GET", "/api/documents")
        if not isinstance(payload, list) or not all(
            isinstance(item, dict) for item in payload
        ):
            raise InvalidApiResponseError(
                "The backend returned an invalid document list."
            )
        documents = [dict(item) for item in payload]
        for document in documents:
            _validate_document(document)
        return documents

    def upload_document(
        self, filename: str, content: bytes, content_type: str | None = None
    ) -> dict[str, Any]:
        """Upload and synchronously index one supported document."""
        files = {
            "file": (
                filename,
                content,
                content_type or "application/octet-stream",
            )
        }
        payload = self._request_json("POST", "/api/documents", files=files)
        return _validated_document_payload(payload)

    def delete_document(self, document_id: str) -> None:
        """Delete one document and all of its indexed chunks."""
        self._request("DELETE", f"/api/documents/{document_id}")

    def reindex_document(self, document_id: str) -> dict[str, Any]:
        """Ask the backend to reload and replace one document's chunks."""
        payload = self._request_json("POST", f"/api/documents/{document_id}/reindex")
        return _validated_document_payload(payload)

    def chat(
        self,
        question: str,
        *,
        conversation_id: str | None = None,
        document_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Submit a grounded chat request using the existing public endpoint."""
        payload: dict[str, Any] = {"question": question}
        if conversation_id:
            payload["conversation_id"] = conversation_id
        if document_ids:
            payload["document_ids"] = list(document_ids)

        response_payload = self._request_json("POST", "/api/chat", json=payload)
        if not isinstance(response_payload, dict):
            raise InvalidApiResponseError(
                "The backend returned an invalid chat response."
            )
        answer = response_payload.get("answer")
        sources = response_payload.get("sources")
        returned_conversation_id = response_payload.get("conversation_id")
        if not isinstance(answer, str) or not isinstance(sources, list):
            raise InvalidApiResponseError(
                "The backend returned an invalid chat response."
            )
        if not all(isinstance(source, dict) for source in sources):
            raise InvalidApiResponseError(
                "The backend returned invalid source citations."
            )
        if returned_conversation_id is not None and not isinstance(
            returned_conversation_id, str
        ):
            raise InvalidApiResponseError(
                "The backend returned an invalid conversation ID."
            )
        return dict(response_payload)

    def _request_json(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self._request(method, path, **kwargs)
        try:
            return response.json()
        except ValueError as exc:
            raise InvalidApiResponseError(
                "The backend returned a response that was not valid JSON."
            ) from exc

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.ConnectError as exc:
            raise ApiUnavailableError(
                "Cannot reach the FastAPI server. Confirm that the backend is running."
            ) from exc
        except httpx.TimeoutException as exc:
            raise ApiTimeoutError(
                "The backend took too long to respond. Indexing and model calls "
                "can take several seconds."
            ) from exc
        except httpx.RequestError as exc:
            raise ApiUnavailableError(
                "The request could not reach the FastAPI server."
            ) from exc

        if response.is_error:
            raise ApiRequestError(
                _extract_error_message(response),
                status_code=response.status_code,
            )
        return response


def _validated_document_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise InvalidApiResponseError("The backend returned invalid document metadata.")
    document = dict(payload)
    _validate_document(document)
    return document


def _validate_document(document: Mapping[str, Any]) -> None:
    required_strings = ("document_id", "filename", "status")
    if any(not isinstance(document.get(key), str) for key in required_strings):
        raise InvalidApiResponseError("The backend returned invalid document metadata.")
    chunk_count = document.get("chunk_count", 0)
    if not isinstance(chunk_count, int):
        raise InvalidApiResponseError(
            "The backend returned an invalid document chunk count."
        )


def _extract_error_message(response: httpx.Response) -> str:
    fallback = f"The backend returned HTTP {response.status_code}."
    try:
        payload = response.json()
    except ValueError:
        return fallback
    if not isinstance(payload, dict):
        return fallback

    detail = payload.get("detail")
    if isinstance(detail, str) and detail.strip():
        return detail
    if isinstance(detail, list):
        messages = [
            str(item.get("msg"))
            for item in detail
            if isinstance(item, dict) and item.get("msg")
        ]
        if messages:
            return " ".join(messages)
    return fallback
