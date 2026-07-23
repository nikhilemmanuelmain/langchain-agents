# Streamlit Manual Test UI

This lightweight UI exercises the same public FastAPI endpoints that another
frontend would use. It does not import or duplicate retrieval, embedding,
conversation, or answer-generation logic.

## Prerequisites

From the repository root, install the locked project dependencies:

```bash
uv sync
```

Create the backend environment file if it does not exist:

```bash
cp .env.example .env
```

Set a valid `OPENAI_API_KEY` in `.env`. Never commit that file.

## Run both services

Open one terminal in the repository root and start FastAPI:

```bash
uv run uvicorn app.main:app --reload
```

The backend is available at `http://localhost:8000`. Its interactive API
documentation is at `http://localhost:8000/docs`.

Open a second terminal in the same repository and start Streamlit:

```bash
uv run streamlit run test_ui/streamlit_app.py
```

Streamlit normally opens `http://localhost:8501` automatically.

## Use another backend URL

The UI reads only the backend base URL from `CHATBOT_API_URL`. For example:

```bash
CHATBOT_API_URL=http://127.0.0.1:9000 \
  uv run streamlit run test_ui/streamlit_app.py
```

The default is:

```text
http://localhost:8000
```

No API key is read or sent by Streamlit. Provider credentials remain in the
FastAPI process.

## Manual test flow

1. Confirm the UI can load the document list.
2. Upload a `.pdf`, `.md`, `.markdown`, or `.txt` file.
3. Confirm its status becomes `indexed` and its chunk count appears.
4. Select the document in the optional filter and ask a supported question.
5. Check that the answer and source citation appear.
6. Ask a context-dependent follow-up and confirm the conversation continues.
7. Clear the conversation and confirm the local chat history resets.
8. Reindex the document.
9. Delete it and confirm it disappears from the list.
10. Stop FastAPI temporarily to confirm the UI reports that it is unavailable.

Uploads and reindexing are synchronous in the current backend, so larger files
can take several seconds while OpenAI embeddings are generated. The UI shows a
loading indicator during those requests.
