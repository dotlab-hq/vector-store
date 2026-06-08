from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from apps.api.dependencies import (
    check_service_health,
    get_scheduler,
    get_workflow,
    init_dependencies,
    init_vector_store_scheduler,
    rebuild_bm25,
)
from apps.api.middleware import AuthMiddleware
from apps.api.routes import documents, files, ingestion, query, vector_stores
from src.database import Base, engine
from src.observability.logging import get_logger, setup_logging

from src.config import settings

STATIC_DIR = Path(__file__).resolve().parent / "static"

setup_logging()
logger = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Validate critical settings at startup
    if not settings.openai_api_key:
        logger.warning("OPENAI_API_KEY not set — LLM calls will fail")

    # Startup: create tables and initialize workflow dependencies
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("database_connected")
    except Exception as e:
        logger.warning("database_connection_failed", error=str(e))

    init_vector_store_scheduler()
    get_scheduler().start()
    try:
        init_dependencies()
        await rebuild_bm25()
        get_workflow()  # compile workflow once at startup
        await check_service_health()
    except Exception as e:
        logger.warning("dependency_init_failed", error=str(e))
        # scheduler still started — stores will be missing but vector store
        # file processing won't crash; the worker checks for None stores
    try:
        yield
    finally:
        await get_scheduler().stop()


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
