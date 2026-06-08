# Agentic RAG

Enterprise-grade Retrieval-Augmented Generation (RAG) platform targeting 10M+
documents, built on LangGraph, LangChain, FastAPI, PostgreSQL, Redis, Qdrant,
and S3-compatible storage.

The full design lives in [`AGENTS.md`](AGENTS.md), [`PLAN.md`](PLAN.md), and
[`ARCHITECTURE.md`](ARCHITECTURE.md).

## Layout

```
.
├── apps/
│   ├── api/        # FastAPI service (HTTP entry point)
│   └── worker/     # Background worker (task processing, cron jobs)
├── src/            # All business logic
│   ├── ingestion/  # Loaders, chunking, metadata extraction
│   ├── indexing/   # Qdrant, BM25 stores
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
├── Dockerfile      # Multi-stage build (API + worker)
├── entrypoint.sh   # Role dispatch: api | worker | smoke
└── docker-compose.yml
```

## Prerequisites

- Python **3.14+**
- [uv](https://docs.astral.sh/uv/) (the only supported package manager)
- A populated `.env` (copy from `.env.example`)

```bash
cp .env.example .env
# then fill in OPENAI_API_KEY, DATABASE_URL, QDRANT_URL, etc.
```

## Install

```bash
uv sync
```

## Run locally

All three entry points use `uv run` so the venv and `PYTHONPATH` resolve.

### 1. Hello-world LLM call (smoke test)

```bash
uv run python -m src.main
```

### 2. FastAPI service (HTTP API)

The API provides query endpoints, file uploads, ingestion, and vector store
management. It creates task rows in PostgreSQL and returns immediately — the
worker handles the heavy processing.

```bash
uv run uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload
```

- Swagger UI → http://localhost:8000/docs
- Healthcheck → http://localhost:8000/health
- OpenAPI schema → http://localhost:8000/openapi.json

**Requires a running worker** to process ingestion tasks. Start the worker in a
separate terminal:

```bash
uv run python -m apps.worker
```

### 3. Background worker

```bash
uv run python -m apps.worker
```

Processes the task queue: document ingestion, text ingestion, document
re-indexing, and vector store file tagging. Recovers stale tasks from crashes
automatically on startup.

## Deploy with Docker

### Build

```bash
docker build -t agentic-rag:latest .
```

### Run via entrypoint

The image uses `entrypoint.sh` to dispatch between three roles:

```bash
# API (default)
docker run --rm -p 8000:8000 --env-file .env agentic-rag:latest

# Worker
docker run --rm --env-file .env agentic-rag:latest worker

# Smoke test
docker run --rm --env-file .env agentic-rag:latest smoke
```

### Run with docker-compose (local development)

The compose file starts a local PostgreSQL, Qdrant, Redis, the API, and a
worker. All other services (S3, OpenAI, LangSmith) are cloud-hosted.

```bash
# Start everything
docker compose up --build -d

# Tail logs
docker compose logs -f api worker

# Stop
docker compose down
```

### Architecture: API vs Worker

| Component | Role | Ports | Lifecycle |
|-----------|------|-------|-----------|
| **API** | HTTP entry point — serves queries, file uploads, ingestion requests | 8000 | Runs until workflow ends |
| **Worker** | Processes the `processing_tasks` queue (ingestion, indexing, VS tagging) + cron sweep | none | Runs until workflow ends; recovers on restart |

The worker is the only process that does heavy work (S3 download, parsing,
chunking, embedding, Qdrant/BM25 indexing). The API creates task rows in
PostgreSQL and returns immediately (200). The worker picks them up, and stale
tasks are automatically recovered on startup or via the cron sweep.

## GitHub Actions workflow

The `Run` workflow (`.github/workflows/run.yml`) runs the worker directly on
the runner via `uv` — no Docker build, no image push. Triggered on a cron
schedule (every 5 hours) or manual dispatch, with a 6-hour timeout.

Each run:
1. Installs Python 3.14 + dependencies via `uv sync`
2. Runs `python -m apps.worker` for up to 6 hours
3. The worker polls `processing_tasks`, runs ingestion/indexing, and executes
   the `VectorStoreCron` sweep (retry failed files, expire stores, recover
   stale tasks)
4. When the job ends the runner is destroyed — no cleanup needed

Stale tasks are recovered automatically on startup and on each cron tick.

### Required GitHub Secrets

| Secret | Purpose |
|--------|---------|
| `DATABASE_URL` | PostgreSQL connection string (`postgresql://...?sslmode=require`) |
| `REDIS_URL` | Redis connection string (`redis://default:pass@host:port`) |
| `QDRANT_URL` | Qdrant server URL (e.g. `https://xyz.qdrant.io:6333`) |
| `QDRANT_API_KEY` | Qdrant Cloud API key |
| `S3_ACCESS_KEY` | AWS / S3-compatible access key |
| `S3_SECRET_KEY` | AWS / S3-compatible secret key |
| `S3_BUCKET` | S3 bucket name (default: `rag`) |
| `OPENAI_API_KEY` | OpenAI API key |
| `OPENAI_BASE_URL` | (optional) Custom endpoint if using a proxy |
| `AUTH_SECRET` | API auth secret |
| `LANGSMITH_API_KEY` | (optional) LangSmith tracing |
| `LANGSMITH_PROJECT` | (optional) LangSmith project name |
| `LANGSMITH_TRACING` | (optional) Set `true` to enable |

## Environment variables

All settings flow through `src/config/settings.py` (pydantic-settings). See
`.env.example` for the full list; the most important:

| Variable | Purpose | Default |
|----------|---------|---------|
| `OPENAI_API_KEY` | Required for chat + embeddings | — |
| `DATABASE_URL` | PostgreSQL connection string | — |
| `REDIS_URL` | Cache + session store | — |
| `QDRANT_URL` | Vector store server (empty = in-memory dev mode) | — |
| `S3_ENDPOINT` | MinIO endpoint (empty = real AWS S3) | — |
| `S3_BUCKET` | S3 bucket | `vector-store` |
| `TASK_WORKER_CONCURRENCY` | Worker parallel task slots | `5` |
| `TASK_WORKER_POLL_INTERVAL_S` | Seconds between poll cycles | `2.0` |
| `TASK_WORKER_LEASE_MINUTES` | Stale task recovery threshold | `15` |

## Coding Rules

- Business logic lives in `src/`, never in `apps/`.
- Every external service has an adapter behind a protocol (e.g. `VectorStore`).
- All state and I/O is typed with Pydantic models.
- No circular imports between `src/` modules.

## License

MIT
