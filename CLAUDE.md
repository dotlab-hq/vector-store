# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Enterprise-grade RAG (Retrieval-Augmented Generation) platform targeting 10M+ documents. Currently in early stage — only `src/main.py` (hello-world LLM call) and `src/llm/openai.py` (ChatOpenAI wrapper) are implemented. The full architecture is specified in `AGENTS.md` and `PLAN.md`.

## Development Commands

```bash
# Package manager: uv (not pip)
uv sync                    # install dependencies
uv run python -m src.main  # run the app

# Adding dependencies
uv add <package>
uv add --dev <package>     # dev-only
```

No linting, formatting, or test commands are configured yet. Per `AGENTS.md`, the project will use Ruff, Pyright, and Black once set up.

## Architecture

Technology stack (planned): Python 3.14, LangGraph, LangChain, LangSmith, FastAPI, PostgreSQL, Redis, FAISS, MinIO/S3.

Current file structure:
- `src/main.py` — entry point
- `src/llm/openai.py` — configures `ChatOpenAI` (gpt-5, temp=0.7), prompts for `OPENAI_API_KEY` if unset
- `src/llm/__init__.py` — re-exports the `llm` instance

Planned module layout under `src/`: `ingestion/`, `indexing/`, `retrieval/`, `graph/`, `agents/`, `generation/`, `memory/`, `evaluation/`, `observability/`, `database/`, `config/`, `shared/`.

## Coding Rules (from AGENTS.md)

- No business logic in `apps/` — keep it in `src/` modules
- All external services require adapter classes (e.g., `VectorStoreAdapter`, `LLMAdapter`)
- Never access FAISS directly — use the `VectorStore` protocol abstraction
- All retrieval operations must be traceable (LangSmith)
- Prompts live in `src/generation/prompts/` only
- Use Pydantic models for all state/schema definitions
- Use a `Settings` class (pydantic-settings) for environment variables, not raw `os.environ`
- No circular imports between `src/` modules
- Type hints are mandatory on all function signatures
- Test directories: `tests/unit/`, `tests/integration/`, `tests/evals/`
