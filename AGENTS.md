# Documentation Chatbot — Project Instructions

## Project Goal

Build a modular Retrieval-Augmented Generation application using Python, FastAPI, LangChain, OpenAI, and Chroma.

The chatbot will be integrated into an existing application and must answer user questions using only the documentation indexed by the system.

The application must:

1. Load documentation.
2. Extract and normalize document content.
3. Split documents into searchable chunks.
4. Generate embeddings.
5. Store embeddings in a vector database.
6. Retrieve relevant chunks for each question.
7. Generate an answer grounded only in retrieved documentation.
8. Return citations containing the source filename and page or section.
9. Clearly state when the documentation does not contain an answer.

This project must be developed module by module. Do not implement future modules unless explicitly requested.

---

## Primary Use Case

A user opens a chatbot inside an application and asks:

> How can I reset my password?

The backend searches the indexed documentation and returns:

```json
{
  "answer": "You can reset your password from the Account Settings page.",
  "sources": [
    {
      "document_id": "user-guide",
      "filename": "user-guide.pdf",
      "page": 12
    }
  ]
}
```

When the information is unavailable, return:

```json
{
  "answer": "I could not find this information in the available documentation.",
  "sources": []
}
```

The model must never invent an answer that is not supported by retrieved documentation.

---

## Technology Stack

Use:

* Python 3.12 or a currently supported stable Python version
* FastAPI
* Uvicorn
* Pydantic
* LangChain
* `langchain-openai`
* `langchain-text-splitters`
* `langchain-chroma`
* ChromaDB
* PyPDF or another lightweight PDF loader supported by LangChain
* Pytest
* Ruff
* MyPy where practical
* `python-dotenv` or Pydantic settings for environment configuration

Do not use LangGraph, agents, multi-agent workflows, rerankers, or hybrid search in the initial modules.

Use a simple deterministic RAG pipeline:

```text
Question
   ↓
Retriever
   ↓
Relevant document chunks
   ↓
Prompt containing question and context
   ↓
Chat model
   ↓
Grounded answer with sources
```

---

## Model Configuration

Never hardcode model names or API keys.

Read these values from environment variables:

```text
OPENAI_API_KEY
OPENAI_CHAT_MODEL
OPENAI_EMBEDDING_MODEL
```

Provide safe example values in `.env.example`, but do not place real secrets in the repository.

The implementation should allow the model provider to be replaced later without changing API routes or business logic.

---

## Architecture Rules

Maintain clear separation between:

* API routes
* request and response schemas
* configuration
* document loading
* document chunking
* document indexing
* vector storage
* retrieval
* answer generation

API routes must not contain the complete RAG implementation.

Use dependency injection or factory functions where it keeps components replaceable and testable.

Avoid unnecessary abstraction. Create interfaces only where they provide a realistic replacement boundary.

---

## Proposed Project Structure

```text
documentation-chatbot/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── health.py
│   │   ├── chat.py
│   │   └── documents.py
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── chat.py
│   │   └── documents.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loaders.py
│   │   ├── splitter.py
│   │   └── indexer.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_store.py
│   │   └── retriever.py
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── prompts.py
│   │   └── rag_service.py
│   │
│   └── services/
│       ├── __init__.py
│       └── document_service.py
│
├── data/
│   ├── documents/
│   └── chroma/
│
├── tests/
│   ├── test_health.py
│   ├── test_chat.py
│   ├── test_loaders.py
│   ├── test_splitter.py
│   ├── test_retriever.py
│   └── test_rag_service.py
│
├── .env.example
├── .gitignore
├── AGENTS.md
├── pyproject.toml
└── README.md
```

The structure may be adjusted if the existing repository already follows a clear convention. Do not reorganize unrelated existing code.

---

# Development Modules

## Module 1 — Backend Foundation

Create the FastAPI application with:

```text
GET /health
POST /api/chat
```

Initial request:

```json
{
  "question": "How can I reset my password?"
}
```

Initial temporary response:

```json
{
  "answer": "The chat API is connected.",
  "sources": []
}
```

Requirements:

* Validate that the question is not empty.
* Add centralized configuration.
* Add CORS configuration from environment settings.
* Add basic exception handling.
* Add API tests.
* Add startup instructions to the README.
* Do not add LangChain or document retrieval logic yet.

Module 1 is complete when the server starts, tests pass, and `/api/chat` returns the temporary response.

---

## Module 2 — Document Loading

Support these file types initially:

* PDF
* Markdown
* Plain text

Convert each file into LangChain `Document` objects.

Each document must include metadata where available:

```python
{
    "document_id": "stable-document-id",
    "filename": "user-guide.pdf",
    "source": "data/documents/user-guide.pdf",
    "page": 12,
    "file_type": "pdf"
}
```

Requirements:

* Reject unsupported file types clearly.
* Preserve page numbers for PDFs.
* Use stable document identifiers.
* Keep loading logic independent from FastAPI.
* Add unit tests with small fixtures.
* Do not generate embeddings yet.

Module 2 is complete when documents can be loaded and their text and metadata can be inspected through tests or a small CLI script.

---

## Module 3 — Document Chunking

Use `RecursiveCharacterTextSplitter` initially.

Starting configuration:

```text
chunk_size = 800
chunk_overlap = 120
```

Make both settings configurable.

Every chunk must preserve the original document metadata and add:

```python
{
    "chunk_id": "stable-chunk-id",
    "chunk_index": 0
}
```

Requirements:

* Do not create empty chunks.
* Produce stable chunk IDs where practical.
* Preserve filename, source, document ID, and page number.
* Add tests for chunk size, metadata, overlap, and empty content.

Module 3 is complete when loaded documents are converted into identifiable, traceable chunks.

---

## Module 4 — Embeddings and Vector Search

Use OpenAI embeddings through `langchain-openai`.

Use persistent local Chroma storage during development.

The vector-store implementation must support:

* adding chunks
* searching chunks
* deleting chunks by document ID where supported
* preventing or replacing duplicate document indexing
* persistence between application restarts

Provide a retrieval method similar to:

```python
search(
    query: str,
    top_k: int = 4,
    document_ids: list[str] | None = None
) -> list[Document]
```

Initially use similarity search with `top_k = 4`.

Do not add answer generation yet.

Module 4 is complete when a natural-language query retrieves the expected documentation chunks.

---

## Module 5 — Grounded RAG Answers

Connect retrieval to the chat model.

The system prompt must enforce these rules:

```text
You are a documentation assistant.

Answer the user's question using only the supplied documentation context.

Do not use outside knowledge.

If the supplied context does not contain enough information, say:
"I could not find this information in the available documentation."

Do not claim that a source supports an answer unless the answer is directly
supported by that source.

Keep the answer clear and concise.
```

Return:

```json
{
  "answer": "Answer grounded in the documentation.",
  "sources": [
    {
      "document_id": "user-guide",
      "filename": "user-guide.pdf",
      "page": 12,
      "chunk_id": "user-guide-page-12-chunk-2"
    }
  ]
}
```

Requirements:

* Remove duplicate sources.
* Return only sources actually supplied to the model.
* Handle unavailable or malformed model responses.
* Log retrieval count and processing errors without logging secrets.
* Add tests using mocked model and embedding dependencies.
* Add at least one manual end-to-end test.

Module 5 is complete when the chatbot answers documentation questions, refuses unsupported questions, and returns traceable sources.

---

## Module 6 — Document Management

Add:

```text
POST   /api/documents
GET    /api/documents
GET    /api/documents/{document_id}
DELETE /api/documents/{document_id}
POST   /api/documents/{document_id}/reindex
```

Track:

```json
{
  "document_id": "user-guide",
  "filename": "user-guide.pdf",
  "checksum": "sha256-value",
  "status": "indexed",
  "chunk_count": 42,
  "indexed_at": "ISO-8601 timestamp"
}
```

Required statuses:

```text
pending
processing
indexed
failed
```

Requirements:

* Validate upload size and extension.
* Sanitize filenames.
* Use checksums to detect unchanged duplicate files.
* Delete associated vector chunks when a document is deleted.
* Return useful indexing errors.
* Do not expose local filesystem paths through public responses.

---

## Module 7 — Conversational Follow-Ups

Add conversation IDs and limited chat history.

Example:

```text
User: How many annual leave days are provided?
Assistant: Employees receive 26 annual leave days.
User: Can they carry them into next year?
```

The second question must be interpreted in the context of the first exchange.

Requirements:

* Keep each conversation isolated.
* Do not mix user sessions.
* Rewrite context-dependent questions into standalone retrieval queries.
* Limit stored history.
* Keep retrieval grounded in documentation.
* Do not allow previous assistant answers to become documentary evidence.

Do not implement this module using LangGraph unless its use becomes justified by a concrete requirement.

---

## Module 8 — Evaluation and Production Readiness

Create a test dataset containing:

```json
{
  "question": "How long is the trial period?",
  "expected_source": "employee-guide.pdf",
  "expected_answer_terms": ["six months"],
  "answerable": true
}
```

Evaluate separately:

1. Retrieval accuracy
2. Answer correctness
3. Groundedness
4. Citation correctness
5. Unsupported-question refusal

Production considerations:

* authentication
* user and tenant isolation
* document permissions
* metadata filtering
* rate limiting
* structured logging
* request IDs
* monitoring
* secure secret handling
* prompt-injection resistance
* maximum document size
* maximum question length
* timeout handling
* database migration from Chroma to PostgreSQL with pgvector if needed

---

# API Conventions

Use Pydantic request and response models.

Example chat request:

```python
class ChatRequest(BaseModel):
    question: str
    conversation_id: str | None = None
    document_ids: list[str] | None = None
```

Example source response:

```python
class SourceReference(BaseModel):
    document_id: str
    filename: str
    page: int | None = None
    section: str | None = None
    chunk_id: str
```

Example chat response:

```python
class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceReference]
```

Use appropriate HTTP status codes and clear error responses.

---

# Coding Standards

* Add type hints to public functions.
* Prefer small focused functions.
* Use descriptive names.
* Avoid global mutable state.
* Do not silently catch exceptions.
* Do not log API keys, document contents, or private user data.
* Add docstrings where behavior is not obvious.
* Use async code only where it provides a real benefit.
* Do not mix synchronous and asynchronous APIs carelessly.
* Keep configuration outside business logic.
* Keep external integrations replaceable.
* Avoid deprecated LangChain imports and APIs.
* Verify imports against the currently installed package versions.
* Prefer current provider-specific LangChain packages.
* Pin compatible dependency versions after confirming the application works.

---

# Testing Standards

Each module must include tests before it is considered complete.

Tests must cover:

* successful behavior
* validation errors
* dependency failures
* unsupported inputs
* missing documentation answers
* source metadata preservation

Mock paid model calls in automated tests.

Do not require a real OpenAI API key for the normal unit-test suite.

Mark real API tests separately as integration tests.

---

# Development Workflow

For every requested module:

1. Inspect the existing repository.
2. Explain the files that will be created or changed.
3. Identify assumptions.
4. Implement only the requested module.
5. Add or update tests.
6. Run formatting, linting, type checks, and tests.
7. Fix failures caused by the change.
8. Summarize:

   * files changed
   * important decisions
   * commands run
   * test results
   * remaining limitations
9. Stop and wait for the next module request.

Do not proceed automatically to another module.

---

# Scope Restrictions

Do not implement these features unless explicitly requested:

* autonomous agents
* multi-agent systems
* web search
* general internet answers
* model fine-tuning
* OCR
* image understanding
* voice input
* reranking
* hybrid keyword/vector retrieval
* knowledge graphs
* frontend redesign
* cloud deployment
* user authentication
* LangGraph orchestration

The first milestone is a reliable documentation chatbot, not a general-purpose AI assistant.
