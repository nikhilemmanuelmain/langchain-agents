# Building Module 1: A Tested FastAPI Foundation with uv

This tutorial explains how Module 1 of the documentation chatbot was built. At
this stage, the backend accepts a question and proves that the API connection
works. It does not load documents, retrieve context, call an AI model, or create
embeddings. Those responsibilities belong to later modules.

The goal is deliberately modest: establish a clean, configurable, and tested
HTTP boundary before adding the expensive and more complicated RAG pipeline.
That gives future modules a known-good foundation and makes bugs easier to
isolate.

## Libraries and tools used

### Python 3.12+

Python is the implementation language. Public functions have type hints, and
the project declares Python 3.12 or newer in `pyproject.toml`. The code uses
modern type syntax such as `Settings | None` and `list[str]`.

### uv

`uv` manages dependency resolution, the lockfile, the project environment, and
command execution. It replaces the manual combination of `python -m venv`,
`pip`, and hand-maintained requirements files.

- `pyproject.toml` declares direct dependencies.
- `uv.lock` records exact resolved versions for repeatable installations.
- `uv sync` makes the local environment match the lockfile.
- `uv run ...` runs commands inside that environment.

### FastAPI

FastAPI is the web framework. It provides routing, middleware, dependency
integration, exception-handler registration, OpenAPI generation, and automatic
request/response validation through Pydantic.

### Uvicorn

Uvicorn is the ASGI server that runs the FastAPI application. FastAPI defines
the application; Uvicorn opens the network port and sends HTTP requests to it.
The `standard` extra installs useful production-grade event-loop, HTTP parsing,
and file-watching dependencies.

### Pydantic

Pydantic defines and validates the JSON contracts. `ChatRequest` rejects an
empty question, while `ChatResponse` and `HealthResponse` guarantee the shapes
returned by the API. FastAPI also uses these models to build the OpenAPI schema.

### pydantic-settings and python-dotenv

`pydantic-settings` loads typed application settings from environment variables
and, during local development, from `.env`. It uses `python-dotenv` to read the
file. Configuration therefore stays outside route and business logic.

### Starlette middleware

FastAPI is built on Starlette. Module 1 uses Starlette's `CORSMiddleware` through
FastAPI to control which browser origins may call the backend. It also uses
Starlette's JSON response type in centralized exception handlers.

### pytest, HTTPX, and TestClient

Pytest discovers and runs the automated tests. FastAPI's `TestClient` provides
an in-process HTTP client for exercising routes without starting a real server;
the client transport is supplied by HTTPX. Tests therefore verify status codes,
headers, and JSON exactly as an external caller would see them.

### Ruff

Ruff is both the formatter and linter. Formatting makes code layout consistent,
while lint rules catch common errors, unused imports, import-order issues, and
outdated Python patterns.

### Existing LangChain packages

The repository already contained LangChain, provider packages, and
`deepagents` for an earlier weather-agent experiment. Module 1 preserves those
dependencies but does not import or use them. Retrieval, embeddings, Chroma,
OpenAI calls, and agent behavior remain outside this module.

## What we built

The foundation has two endpoints:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Confirms that the API process is available |
| `POST` | `/api/chat` | Validates a question and returns a temporary response |

The code is separated by responsibility:

```text
app/
├── api/          # HTTP routes
├── schemas/      # Pydantic request and response contracts
├── config.py     # Environment-based settings
├── errors.py     # Shared exception handlers
└── main.py       # Application factory and Uvicorn entry point
tests/            # Endpoint, configuration, and error-handler tests
```

This separation keeps the routes small and leaves clear places for future
modules without prematurely implementing them.

The request flow is:

```text
HTTP request
    ↓
CORS middleware
    ↓
FastAPI route matching
    ↓
Pydantic request validation
    ↓
Route function
    ↓
Pydantic response serialization
    ↓
JSON HTTP response
```

## 1. Install the project

The project uses `uv`, so no manual virtual-environment activation is needed.
From the repository root, run:

```bash
uv sync
```

`uv sync` reads `pyproject.toml`, installs the locked runtime and development
dependencies, and manages `.venv` automatically.

Suggested screenshot: the terminal after `uv sync` completes, showing the
resolved and installed packages.

## 2. Configure the environment

Create a local configuration file from the safe example:

```bash
cp .env.example .env
```

The example contains:

```dotenv
APP_NAME=Documentation Chatbot API
APP_ENV=development
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
```

`pydantic-settings` reads these values when the application starts.
`CORS_ALLOWED_ORIGINS` accepts a comma-separated list, and whitespace around
each origin is removed. Leave the value empty if no browser origin should be
allowed.

The `Settings` class provides defaults, so the server can start without a local
`.env`. Its computed `cors_origins` property converts the environment string
into the list expected by CORS middleware. `get_settings()` is cached so normal
application startup does not repeatedly parse the environment.

The real `.env` file is ignored by Git. `.env.example` remains tracked so every
developer knows which settings are available without exposing secrets.

## 3. Define the API schemas

`app/schemas/chat.py` owns the chat contract. The request contains one field:

```python
class ChatRequest(BaseModel):
    question: str
```

A field validator strips surrounding whitespace. If nothing remains, it raises
a validation error with the message `Question must not be empty.`. Returning the
stripped value also means downstream code receives a normalized question.

The temporary response schema contains:

```python
class ChatResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
```

The source list remains empty in Module 1. A richer source-reference model will
only be justified when document retrieval is implemented.

`HealthResponse` limits `status` to the literal value `"ok"`. This is a small
contract, but it prevents the implementation and API documentation from
silently drifting apart.

## 4. Add focused route modules

The health router exposes `GET /health` and returns `{"status": "ok"}`. Health
checks are intentionally cheap and do not call external AI services.

The chat router uses the `/api` prefix and exposes `POST /chat`. FastAPI parses
the incoming JSON into `ChatRequest` before calling the function. The route then
returns the required temporary value:

```python
ChatResponse(answer="The chat API is connected.", sources=[])
```

Keeping the placeholder explicit is important. A connectivity milestone should
not accidentally become an untested retrieval implementation.

## 5. Understand the application entry point

The command used to start the API is:

```bash
uv run uvicorn app.main:app --reload
```

The final argument has this meaning:

```text
app.main:app
│   │    └── FastAPI object named "app"
│   └─────── Python module main.py
└─────────── Python package app/
```

`app/main.py` also exposes `create_app()`. The factory lets tests create an
application with specific settings, which is especially useful when checking
CORS behavior.

Inside the factory, the application is assembled in four steps:

1. Load the supplied settings or the cached environment settings.
2. Create the `FastAPI` object with the configured application name.
3. Register CORS middleware and centralized exception handlers.
4. Include the health and chat routers.

The module-level line `app = create_app()` creates the object Uvicorn imports.
This is the FastAPI equivalent of an executable application's main entry point.

Suggested screenshot: the terminal showing “Application startup complete” and
the Uvicorn URL.

## 6. Configure CORS

CORS is a browser security mechanism. A backend may work perfectly through
`curl` yet be blocked when JavaScript from another origin calls it. An origin is
the combination of scheme, host, and port, so `http://localhost:3000` and
`http://localhost:5173` are different origins.

Module 1 passes the parsed environment list to `CORSMiddleware`. Methods and
headers are allowed broadly, but origins are explicit. This makes frontend
development configurable without hardcoding a particular deployment URL.

For example:

```dotenv
CORS_ALLOWED_ORIGINS=https://docs.example.com,https://admin.example.com
```

Avoid using `*` with credentials in a real deployment. Explicit origins are
easier to audit and behave correctly with credentialed browser requests.

## 7. Centralize error responses

`app/errors.py` defines handlers for three error categories:

- `RequestValidationError` produces HTTP `422` and `validation_error`;
- `HTTPException` preserves its intended status and produces `http_error`;
- an unexpected `Exception` produces HTTP `500` and
  `internal_server_error`.

The validation details pass through FastAPI's `jsonable_encoder`. Pydantic
errors may contain Python exception objects in their context, and raw JSON
serialization cannot encode those objects safely.

Unexpected errors are logged with request method and path for diagnosis, while
the client receives only a generic message. This avoids leaking internal stack
traces or private details through the public API.

## 8. Try the endpoints

Open another terminal while Uvicorn is running.

Check the process health:

```bash
curl http://127.0.0.1:8000/health
```

Expected result:

```json
{"status":"ok"}
```

Send a valid question:

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"How can I reset my password?"}'
```

Expected result:

```json
{
  "answer": "The chat API is connected.",
  "sources": []
}
```

The answer is intentionally fixed. It proves the frontend-to-backend contract
before retrieval and generation logic are introduced.

Now verify validation with a blank question:

```bash
curl -i -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"   "}'
```

The API returns HTTP `422` with a centralized `validation_error` response.
Trimming and validation happen in the Pydantic request schema, so every caller
gets the same rule.

Suggested screenshots:

1. The successful `/health` response.
2. The successful `/api/chat` placeholder response.
3. The HTTP `422` response for a whitespace-only question.

You can also visit `http://127.0.0.1:8000/docs` to use FastAPI's interactive
OpenAPI page. A useful screenshot is the expanded `POST /api/chat` operation
showing its request and response schemas.

An unexpected failure receives a safe and predictable public response:

```json
{
  "error": "internal_server_error",
  "detail": "An unexpected error occurred."
}
```

## 9. Understand the automated tests

The test suite is split by responsibility:

- `test_health.py` verifies the status code and exact health JSON.
- `test_chat.py` verifies the placeholder response, a missing question, and
  empty or whitespace-only questions.
- `test_config.py` verifies comma-separated origin parsing and a real CORS
  preflight response.
- `test_errors.py` verifies that an unexpected exception produces HTTP `500`
  without exposing its private message.

Tests use the application exactly at its HTTP boundary where practical. The
configuration test calls `create_app(custom_settings)`, demonstrating why the
application factory is useful: it changes configuration without mutating the
developer's environment.

## 10. Format, lint, and test

Run the same quality checks used to verify this module:

```bash
uv run ruff format .
uv run ruff check .
uv run pytest
```

At completion, Ruff reported no lint errors and pytest reported:

```text
9 passed
```

The tests cover:

- a successful health check;
- the exact temporary chat response;
- missing, empty, and whitespace-only questions;
- parsing configured CORS origins;
- a successful CORS preflight request;
- hiding internal details in unexpected-error responses.

Suggested screenshot: the final terminal output showing Ruff passing and all
nine tests passing.

## 11. Complete command sequence

For a new checkout, the full Module 1 workflow is:

```bash
# Install exactly what the lockfile specifies
uv sync

# Create local configuration
cp .env.example .env

# Format and verify the code
uv run ruff format .
uv run ruff check .
uv run pytest

# Start the API
uv run uvicorn app.main:app --reload
```

In a second terminal:

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"How can I reset my password?"}'
```

When changing dependencies, use `uv add <package>` or `uv remove <package>` so
`pyproject.toml` and `uv.lock` remain synchronized.

## 12. Common problems

### `uv: command not found`

Install `uv` using the official installation method for your operating system,
then open a new terminal and run `uv --version`.

### `Address already in use`

Another process is using port 8000. Stop that process or choose another port:

```bash
uv run uvicorn app.main:app --reload --port 8001
```

### The frontend gets a CORS error

Add the frontend's exact scheme, hostname, and port to
`CORS_ALLOWED_ORIGINS`, restart Uvicorn, and try again. Do not include URL paths
such as `/chat` in an origin.

### A blank question returns 422

This is expected. The API treats missing, empty, and whitespace-only questions
as invalid input. Send at least one non-whitespace character.

### Changes are not visible

Use the `--reload` flag during local development. Do not normally use automatic
reload in production.

## What intentionally comes later

This module creates only the backend foundation. It contains no document
loader, text splitter, LangChain retrieval pipeline, embedding model, Chroma
database, OpenAI request, or generated answer. Keeping that boundary explicit
makes Module 1 independently testable and gives later modules a stable API
contract to build upon.
