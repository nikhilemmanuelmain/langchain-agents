"""Lightweight Streamlit UI for manually exercising the public FastAPI API."""

import os
from datetime import datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import TYPE_CHECKING, Any

import streamlit as st

if TYPE_CHECKING:
    from test_ui.api_client import ChatbotApiClient, ChatbotApiError
else:
    client_path = Path(__file__).with_name("api_client.py")
    client_spec = spec_from_file_location("streamlit_api_client", client_path)
    if client_spec is None or client_spec.loader is None:
        raise ImportError(f"Could not load the API client from {client_path}.")
    client_module = module_from_spec(client_spec)
    client_spec.loader.exec_module(client_module)
    ChatbotApiClient = client_module.ChatbotApiClient
    ChatbotApiError = client_module.ChatbotApiError

DEFAULT_API_URL = "http://localhost:8000"
SUPPORTED_UPLOAD_TYPES = ["pdf", "md", "markdown", "txt"]


def _initialize_session() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("conversation_id", None)
    st.session_state.setdefault("notice", None)


def _show_notice() -> None:
    notice = st.session_state.pop("notice", None)
    if notice:
        level, message = notice
        getattr(st, level)(message)


def _format_indexed_at(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return parsed.astimezone().strftime("%Y-%m-%d %H:%M")


def _document_label(document: dict[str, Any]) -> str:
    return f"{document['filename']} ({document['document_id']})"


def _render_document_details(document: dict[str, Any]) -> None:
    status = str(document["status"])
    st.markdown(f"**{document['filename']}**")
    st.caption(f"Status: {status}")
    chunk_count = document.get("chunk_count")
    if isinstance(chunk_count, int):
        st.caption(f"Chunks: {chunk_count}")
    indexed_at = _format_indexed_at(document.get("indexed_at"))
    if indexed_at:
        st.caption(f"Indexed: {indexed_at}")
    if status == "failed" and document.get("error"):
        st.error(str(document["error"]))


def _render_document_management(
    client: ChatbotApiClient, documents: list[dict[str, Any]]
) -> None:
    st.sidebar.header("Documents")
    uploaded_file = st.sidebar.file_uploader(
        "Choose documentation",
        type=SUPPORTED_UPLOAD_TYPES,
        help="Supported formats: PDF, Markdown, and plain text.",
    )
    if st.sidebar.button(
        "Upload and Index",
        type="primary",
        disabled=uploaded_file is None,
        width="stretch",
    ):
        if uploaded_file is None:
            st.sidebar.warning("Choose a supported document first.")
        else:
            try:
                with st.spinner("Uploading and indexing document..."):
                    result = client.upload_document(
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type,
                    )
                if result["status"] == "indexed":
                    st.session_state.notice = (
                        "success",
                        f"{result['filename']} was indexed successfully.",
                    )
                else:
                    st.session_state.notice = (
                        "warning",
                        f"{result['filename']} was uploaded with status "
                        f"{result['status']}.",
                    )
                st.rerun()
            except ChatbotApiError as exc:
                st.sidebar.error(f"Upload failed: {exc}")

    if st.sidebar.button("Refresh document list", width="stretch"):
        st.rerun()

    if not documents:
        st.sidebar.info("No documents are indexed yet.")
        return

    st.sidebar.divider()
    for document in documents:
        document_id = str(document["document_id"])
        with st.sidebar.container(border=True):
            _render_document_details(document)
            reindex_column, delete_column = st.columns(2)
            if reindex_column.button(
                "Reindex",
                key=f"reindex-{document_id}",
                width="stretch",
            ):
                try:
                    with st.spinner(f"Reindexing {document['filename']}..."):
                        result = client.reindex_document(document_id)
                    st.session_state.notice = (
                        "success",
                        f"{result['filename']} was reindexed successfully.",
                    )
                    st.rerun()
                except ChatbotApiError as exc:
                    st.sidebar.error(f"Reindexing failed: {exc}")

            if delete_column.button(
                "Delete",
                key=f"delete-{document_id}",
                width="stretch",
            ):
                try:
                    client.delete_document(document_id)
                    st.session_state.notice = (
                        "success",
                        f"{document['filename']} was deleted.",
                    )
                    st.rerun()
                except ChatbotApiError as exc:
                    st.sidebar.error(f"Deletion failed: {exc}")


def _render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        st.caption("No sources were returned.")
        return
    st.markdown("**Sources**")
    for source in sources:
        parts = [str(source.get("filename") or "Unknown document")]
        page = source.get("page")
        section = source.get("section")
        if page is not None:
            parts.append(f"page {page}")
        if section:
            parts.append(str(section))
        st.caption(" — ".join(parts))


def _render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                _render_sources(message.get("sources", []))


def _clear_conversation() -> None:
    st.session_state.messages = []
    st.session_state.conversation_id = None


def _render_chat(client: ChatbotApiClient, documents: list[dict[str, Any]]) -> None:
    indexed_documents = [
        document for document in documents if document["status"] == "indexed"
    ]
    selected_ids = st.multiselect(
        "Optional document filter",
        options=[str(document["document_id"]) for document in indexed_documents],
        format_func=lambda document_id: next(
            _document_label(document)
            for document in indexed_documents
            if document["document_id"] == document_id
        ),
        placeholder="Search all indexed documents",
        disabled=not indexed_documents,
    )
    if not indexed_documents:
        st.info("Upload and index a document before asking grounded questions.")

    if st.button("Clear conversation"):
        _clear_conversation()
        st.rerun()

    _render_chat_history()
    question = st.chat_input(
        "Ask a question about your documentation",
        disabled=not indexed_documents,
    )
    if question is None:
        return
    normalized_question = question.strip()
    if not normalized_question:
        st.warning("Enter a question before sending.")
        return

    st.session_state.messages.append({"role": "user", "content": normalized_question})
    with st.chat_message("user"):
        st.markdown(normalized_question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching documentation and preparing an answer..."):
                response = client.chat(
                    normalized_question,
                    conversation_id=st.session_state.conversation_id,
                    document_ids=selected_ids or None,
                )
            answer = str(response["answer"])
            sources = response["sources"]
            st.markdown(answer)
            _render_sources(sources)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                }
            )
            if response.get("conversation_id"):
                st.session_state.conversation_id = response["conversation_id"]
        except ChatbotApiError as exc:
            st.error(f"The question could not be answered: {exc}")


def main() -> None:
    """Render the manual chatbot testing application."""
    st.set_page_config(page_title="Documentation Assistant", page_icon="📚")
    _initialize_session()
    st.title("Documentation Assistant")
    st.caption("Manual test UI for the existing FastAPI documentation chatbot")
    _show_notice()

    api_url = os.getenv("CHATBOT_API_URL", DEFAULT_API_URL)
    client = ChatbotApiClient(api_url)
    try:
        try:
            documents = client.list_documents()
        except ChatbotApiError as exc:
            documents = []
            st.error(str(exc))
        _render_document_management(client, documents)
        _render_chat(client, documents)
    finally:
        client.close()


if __name__ == "__main__":
    main()
