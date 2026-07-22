# LangChain Agents

This project uses [uv](https://docs.astral.sh/uv/) for Python versions,
dependencies, locking, and command execution.

## Setup

Install the locked dependencies:

```bash
uv sync
```

Run Python or project scripts through `uv` (activation is not required):

```bash
uv run python
uv run python path/to/script.py
```

Run the first weather agent after setting your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key"
uv run python agents/weather_agent.py
```

The example weather tool intentionally returns mock weather, matching the
LangChain quickstart. Replace `get_weather` with a real weather API when you
are ready to fetch live conditions.

Add or remove dependencies with `uv` so that both `pyproject.toml` and
`uv.lock` stay synchronized:

```bash
uv add langchain
uv remove langchain
```

`uv` manages the project environment in `.venv` automatically. Do not create
or maintain it with `python -m venv`, `pip`, or a `requirements.txt` file.
