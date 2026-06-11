import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from apps.api.dependencies import (
    check_service_health,
    get_workflow,
    init_dependencies,
    rebuild_bm25,
)
from apps.api.middleware import AuthMiddleware
from apps.api.routes import documents, files, ingestion, query, vector_stores
from src.database import Base, engine, run_schema_migrations
from src.observability.logging import get_logger, setup_logging

from src.config import settings

STATIC_DIR = Path(__file__).resolve().parent / "static"

setup_logging()
logger = get_logger()

# In-process arq worker — set during lifespan
_arq_worker_task = None
_arq_worker_instance = None


async def _start_inprocess_worker() -> None:
    """Start an arq Worker inside the API process so tasks are consumed
    without requiring a separate ``python -m apps.worker`` process."""
    global _arq_worker_task, _arq_worker_instance

    if not settings.redis_url:
        logger.warning("worker_skipped", reason="REDIS_URL not set")
        return

    from arq import Worker
    from apps.worker.arq_settings import WorkerSettings
    from src.shared.events import get_redis_settings

    try:
        redis_settings = get_redis_settings()
    except Exception as exc:
        logger.warning("worker_redis_connect_failed", error=str(exc))
        return

    _arq_worker_instance = Worker(
        WorkerSettings.functions,
        redis_settings=redis_settings,
        queue_name=WorkerSettings.queue_name,
        max_jobs=WorkerSettings.max_jobs,
        poll_delay=WorkerSettings.poll_delay,
        max_tries=WorkerSettings.max_tries,
        health_check_interval=WorkerSettings.health_check_interval,
    )
    _arq_worker_task = asyncio.create_task(
        _arq_worker_instance.main(), name="inprocess-arq-worker"
    )
    logger.info(
        "inprocess_worker_started",
        max_jobs=WorkerSettings.max_jobs,
        queue=WorkerSettings.queue_name,
    )


async def _stop_inprocess_worker() -> None:
    """Gracefully shut down the in-process arq worker."""
    global _arq_worker_task, _arq_worker_instance
    if _arq_worker_instance is not None:
        try:
            await _arq_worker_instance.close()
        except Exception:
            pass
        _arq_worker_instance = None
    if _arq_worker_task is not None:
        _arq_worker_task.cancel()
        try:
            await _arq_worker_task
        except asyncio.CancelledError:
            pass
        _arq_worker_task = None
    logger.info("inprocess_worker_stopped")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Validate critical settings at startup
    if not settings.openai_api_key:
        logger.warning("OPENAI_API_KEY not set — LLM calls will fail")

    # Startup: create tables and initialize workflow dependencies
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await run_schema_migrations()
        logger.info("database_connected")
    except Exception as e:
        logger.warning("database_connection_failed", error=str(e))

    try:
        init_dependencies()
        await rebuild_bm25()
        get_workflow()  # compile workflow once at startup
        await check_service_health()
    except Exception as e:
        logger.warning("dependency_init_failed", error=str(e))

    # Start the in-process arq worker so tasks are consumed
    await _start_inprocess_worker()

    yield

    # Shutdown: stop worker, close Redis pool
    await _stop_inprocess_worker()
    await close_redis_pool()


app = FastAPI(title="Agentic RAG", version="0.1.0", lifespan=lifespan)
app.add_middleware(AuthMiddleware)


@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    # Loosen CSP for static HTML pages so the playground (Tailwind/Lucide CDN,
    # inline JS) renders correctly. Strict CSP still applies to API routes.
    if request.url.path in ("/", "/util", "/playground"):
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://unpkg.com; "
            "style-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com; "
            "img-src 'self' data:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'"
        )
    else:
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; frame-ancestors 'none'"
        )
    return response


app.include_router(query.router)
app.include_router(files.router)
app.include_router(ingestion.router)
app.include_router(documents.router)
app.include_router(vector_stores.router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "unhandled_exception",
        method=request.method,
        path=str(request.url.path),
        error_type=type(exc).__name__,
        exc_info=True,
    )
    # Never expose internal details to clients
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "detail": "An unexpected error occurred. Please try again later.",
            "path": str(request.url.path),
        },
    )


@app.get("/")
async def landing() -> FileResponse:
    return FileResponse(STATIC_DIR / "landing.html", media_type="text/html")


@app.get("/util")
async def env_util() -> FileResponse:
    return FileResponse(STATIC_DIR / "env-util.html", media_type="text/html")


@app.get("/playground")
async def playground() -> FileResponse:
    return FileResponse(STATIC_DIR / "playground.html", media_type="text/html")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
