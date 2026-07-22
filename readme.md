# Documentation Chatbot

This project uses [uv](https://docs.astral.sh/uv/) for Python versions,
dependencies, locking, and command execution.

Modules 1 and 2 provide the FastAPI backend foundation and standalone document
loading for PDF, Markdown, and plain-text files. The chat route still returns a
fixed connectivity response; chunking, retrieval, embeddings, vector storage,
and model calls are intentionally not implemented yet.

## Setup

Install `uv`, clone the repository, and install the locked dependencies:

```bash
uv sync
```

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

Call the temporary chat endpoint:

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"How can I reset my password?"}'
```

Expected response:

```json
{
  "answer": "The chat API is connected.",
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
