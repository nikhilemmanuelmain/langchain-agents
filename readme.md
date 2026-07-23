# Documentation Chatbot

This project uses [uv](https://docs.astral.sh/uv/) for Python versions,
dependencies, locking, and command execution.

Modules 1 through 5 provide a grounded documentation chatbot: FastAPI, document
loading, traceable chunking, OpenAI embeddings, persistent Chroma retrieval,
and concise answers supported by retrieved source chunks.

## Setup

Install `uv`, clone the repository, and install the locked dependencies:

```bash
uv sync
```

The tracked `.python-version` asks `uv` to use Python 3.13. This keeps local
Chroma compatible with Intel macOS, where newer ONNX Runtime releases no longer
publish x86-64 wheels. `uv` downloads the managed Python version automatically
when needed.

Copy the example configuration and adjust the allowed browser origins when
needed:

```bash
cp .env.example .env
```

`CORS_ALLOWED_ORIGINS` is a comma-separated list. Set it to an empty value to
allow no cross-origin browser clients.

## Run the API

Start the development server from the repository root:

```bash
uv run uvicorn app.main:app --reload
```

The API is then available at `http://127.0.0.1:8000`, with interactive OpenAPI
documentation at `http://127.0.0.1:8000/docs`.

Check health:

```bash
curl http://127.0.0.1:8000/health
```

Call the grounded chat endpoint after indexing documentation:

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"How can I reset my password?"}'
```

Expected response:

```json
{
  "answer": "You can reset your password from the Account Settings page.",
  "sources": [
    {
      "document_id": "guide-e7f436dc4541740f",
      "filename": "guide.md",
      "page": null,
      "section": null,
      "chunk_id": "guide-e7f436dc4541740f-chunk-0-4a696d5847cc"
    }
  ]
}
```

If the indexed documentation does not support an answer, the API returns:

```json
{
  "answer": "I could not find this information in the available documentation.",
  "sources": []
}
```

## Load and inspect documents

Place development documents in `data/documents/`, then inspect one or more
supported files without starting FastAPI:

```bash
uv run python -m app.ingestion.inspect_documents \
  data/documents/guide.pdf \
  data/documents/notes.md
```

Supported extensions are `.pdf`, `.md`, `.markdown`, and `.txt`. Markdown and
text files produce one LangChain `Document`; PDFs produce one `Document` per
page with one-based page metadata. The inspection command prints each
document's metadata, extracted content length, and a short content preview.

Use a different preview size when needed:

```bash
uv run python -m app.ingestion.inspect_documents \
  --preview-length 500 data/documents/notes.txt
```

Document loading is independent from the API. Module 2 does not index files or
send their content to an embedding or chat model.

## Split and inspect document chunks

Load and split one or more documents with the configured defaults:

```bash
uv run python -m app.ingestion.inspect_chunks \
  data/documents/guide.pdf \
  data/documents/notes.md
```

The defaults are configured in `.env`:

```dotenv
CHUNK_SIZE=800
CHUNK_OVERLAP=120
```

You can override them for one inspection run:

```bash
uv run python -m app.ingestion.inspect_chunks \
  --chunk-size 500 \
  --chunk-overlap 75 \
  --preview-length 200 \
  data/documents/notes.txt
```

Each non-empty chunk preserves its source document metadata and adds a
zero-based `chunk_index` plus a stable `chunk_id`. Chunking remains independent
from FastAPI and does not create embeddings or write to a vector database.

## Index and search documents

Module 4 uses OpenAI embeddings and persistent local Chroma storage. Add a real
API key to your local `.env`; never commit it:

```dotenv
OPENAI_API_KEY=your-api-key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-5.6-luna
OPENAI_REQUEST_TIMEOUT_SECONDS=30
OPENAI_MAX_RETRIES=2
CHROMA_PERSIST_DIRECTORY=data/chroma
CHROMA_COLLECTION_NAME=documentation
RETRIEVAL_TOP_K=4
```

Load, chunk, index, and search one or more files from the command line:

```bash
uv run python -m app.retrieval.inspect_search \
  "How can I reset my password?" \
  data/documents/guide.pdf \
  data/documents/notes.md
```

The command prints the matching chunks and their traceability metadata. Chroma
stores its local database under `data/chroma/`, which is ignored by Git.

To limit retrieval to known document IDs, repeat `--document-id`:

```bash
uv run python -m app.retrieval.inspect_search \
  "How can I reset my password?" \
  --document-id guide-e7f436dc4541740f \
  data/documents/guide.md
```

Re-indexing a `document_id` replaces all of its existing chunks, preventing
duplicate and stale chunks.

## Generate a grounded answer

Run the complete load, chunk, index, retrieve, and answer pipeline manually:

```bash
uv run python -m app.generation.inspect_answer \
  "How can I reset my password?" \
  data/documents/guide.md
```

The model receives only the retrieved chunks. Its structured response names the
supporting chunk IDs, and the application rejects citations that were not part
of the retrieved context. Unsupported questions return the fixed refusal text
with no sources.

The manual command also persists the indexed chunks, so the FastAPI `/api/chat`
route can retrieve them afterward. The API accepts an optional document filter:

```json
{
  "question": "How can I reset my password?",
  "document_ids": ["guide-e7f436dc4541740f"]
}
```

The normal test suite uses fake model and embedding dependencies. To run the
paid, real-provider integration test explicitly:

```bash
RUN_OPENAI_INTEGRATION=1 uv run pytest \
  tests/integration/test_rag_e2e.py -m integration
```

## Development checks

Format, lint, and test the project:

```bash
uv run ruff format .
uv run ruff check .
uv run pytest
```

Run Python or project scripts through `uv`; environment activation is not
required:

```bash
uv run python path/to/script.py
```

Add or remove dependencies with `uv` so that both `pyproject.toml` and
`uv.lock` stay synchronized:

```bash
uv add langchain
uv remove langchain
```

`uv` manages the project environment in `.venv` automatically. Do not create
or maintain it with `python -m venv`, `pip`, or a `requirements.txt` file.

## Existing weather-agent example

The earlier learning example remains available and is separate from the
documentation chatbot backend:

```bash
export OPENAI_API_KEY="your-api-key"
uv run python agents/weather_agent.py
```
