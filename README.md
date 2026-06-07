# Agentic RAG

Enterprise-grade Retrieval-Augmented Generation (RAG) platform targeting 10M+
documents, built on LangGraph, LangChain, FastAPI, PostgreSQL, Redis, Qdrant,
FAISS, and S3-compatible storage.

The full design lives in [`AGENTS.md`](AGENTS.md), [`PLAN.md`](PLAN.md), and
[`ARCHITECTURE.md`](ARCHITECTURE.md).

## Layout

```
.
├── apps/
│   ├── api/        # FastAPI service (HTTP entry point)
│   └── worker/     # Background worker (scheduler, cron, vector-store jobs)
├── src/            # All business logic
│   ├── ingestion/  # Loaders, chunking, metadata extraction
│   ├── indexing/   # FAISS, Qdrant, BM25 stores
│   ├── retrieval/  # Dense, sparse, hybrid, KG, reranking
│   ├── graph/      # LangGraph workflow + nodes + routers
│   ├── agents/     # Query agents (LCEL)
│   ├── generation/ # Prompts, context builder, citations, faithfulness
│   ├── memory/     # Session store + semantic cache
│   ├── evaluation/ # RAGAS / custom evaluators
│   ├── observability/  # Structlog + LangSmith
│   ├── database/   # SQLAlchemy models + repositories
│   ├── vector_stores/   # Vector store service layer
│   ├── storage/    # S3 client
│   ├── config/     # pydantic-settings
│   ├── shared/     # Base classes, common types
│   └── llm/        # ChatOpenAI wrapper
├── tests/          # unit / integration / evals
├── docs/
├── pyproject.toml  # uv-managed dependencies
└── Dockerfile
```

## Prerequisites

- Python **3.14+**
- [uv](https://docs.astral.sh/uv/) (the only supported package manager)
- A populated `.env` (copy from `.env.example`)

```bash
cp .env.example .env
# then fill in OPENAI_API_KEY, database URL, etc.
```

## Install

```bash
uv sync
```

This creates a `.venv` and installs runtime + dev dependencies in one step.

## Run

There are three entry points. Run any of them with `uv run` so the venv and
`PYTHONPATH` resolve correctly.

### 1. Hello-world LLM call (smoke test)

```bash
uv run python -m src.main
```

Translates "I love programming." to French via `ChatOpenAI`. Useful for
verifying that `OPENAI_API_KEY` and the `src.llm` wrapper are wired up.

### 2. FastAPI service (HTTP API)

```bash
uv run uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Then:

- Swagger UI → http://localhost:8000/docs
- Healthcheck → http://localhost:8000/health
- OpenAPI schema → http://localhost:8000/openapi.json

### 3. Background worker (scheduler + vector-store jobs)

```bash
uv run python -m apps.worker
```

Runs the cron-driven jobs declared under `src/vector_stores/` (e.g. vector
store maintenance, batch processing).

## Test / Lint (planned)

Per `AGENTS.md`, the project targets Ruff + Pyright once they are wired up.
The dev extras are already declared in `pyproject.toml`:

```bash
uv sync --extra dev
uv run ruff check .
uv run pyright
```

## Run with Docker

The image is multi-stage: a `builder` stage installs dependencies into a uv
managed venv, and a slim `runtime` stage copies only the venv and the
application code. No compilers, no cache, no `.venv` in the final layer.

```bash
# Build
docker build -t vector-store:latest .

# Run the FastAPI service (default CMD)
docker run --rm -p 8000:8000 --env-file .env vector-store:latest

# Run the worker
docker run --rm --env-file .env vector-store:latest worker

# Run the smoke test
docker run --rm --env-file .env vector-store:latest smoke
```

The default `CMD` is `api`, so `docker run vector-store:latest` is equivalent
to `docker run vector-store:latest api`.

## Environment

All settings flow through `src/config/settings.py` (pydantic-settings). Do
**not** read `os.environ` directly anywhere else. See `.env.example` for the
full list; the most important keys:

| Variable                    | Purpose                          |
| --------------------------- | -------------------------------- |
| `OPENAI_API_KEY`            | Required for chat + embeddings   |
| `OPENAI_CHAT_MODEL`         | Default `gpt-4o`                 |
| `OPENAI_EMBEDDING_MODEL`    | Default `text-embedding-3-small` |
| `DATABASE_URL`              | PostgreSQL connection string     |
| `REDIS_URL`                 | Cache + session store            |
| `QDRANT_URL`                | Vector store (Qdrant)            |
| `S3_ENDPOINT` / `S3_BUCKET` | Object storage for raw documents |

## Coding Rules (quick reference)

- Business logic lives in `src/`, never in `apps/`.
- Every external service has an adapter behind a protocol (e.g. `VectorStore`,
  `LLMAdapter`).
- Prompts live in `src/generation/prompts/` only.
- All state and I/O is typed with Pydantic models.
- Retrieval operations are traced via LangSmith.
- No circular imports between `src/` modules.

## License

MIT
