"""Redis-backed event bus using arq for task dispatch.

The API side publishes jobs via ``enqueue_task()``.  The worker side
consumes them via arq's ``WorkerSettings`` (see ``apps/worker/arq_settings.py``).

This module owns:
- The shared ``arq.connections.RedisSettings`` derived from app config.
- A helper ``get_redis_pool()`` / ``close_redis_pool()`` for the API lifespan.
- ``enqueue_task()`` — fire-and-forget job enqueue used by API routes.
- Worker heartbeat tracking — Redis keys with TTL for live worker detection.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from arq import create_pool
from arq.connections import RedisSettings

from src.config import settings

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

WORKER_HEARTBEAT_PREFIX = "worker:heartbeat:"
"""Redis key prefix for worker heartbeat keys."""

WORKER_HEARTBEAT_TTL = 30
"""Seconds after which a worker heartbeat is considered stale."""

# ── Redis connection ─────────────────────────────────────────────────

_redis_settings: RedisSettings | None = None
_pool: Any = None  # arq.arq Redis connection pool


def get_redis_settings() -> RedisSettings:
    """Build arq RedisSettings from the app's ``REDIS_URL``."""
    global _redis_settings
    if _redis_settings is None:
        url = settings.redis_url
        if not url:
            raise RuntimeError("REDIS_URL is not set — cannot enqueue tasks")
        # arq RedisSettings expects a plain redis:// URL
        _redis_settings = RedisSettings.from_dsn(url)
    return _redis_settings


async def get_redis_pool():
    """Return (or create) a shared arq Redis connection pool for the API."""
    global _pool
    if _pool is None:
        _pool = await create_pool(get_redis_settings())
        logger.info("arq_redis_pool_created")
    return _pool


async def close_redis_pool() -> None:
    """Shut down the shared Redis pool (call from API lifespan on shutdown)."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("arq_redis_pool_closed")


# ── Worker heartbeat ────────────────────────────────────────────────


def worker_heartbeat_key(worker_id: str) -> str:
    """Return the Redis key for a worker's heartbeat."""
    return f"{WORKER_HEARTBEAT_PREFIX}{worker_id}"


async def update_worker_heartbeat(worker_id: str) -> None:
    """Write a heartbeat timestamp for this worker with a TTL."""
    pool = await get_redis_pool()
    key = worker_heartbeat_key(worker_id)
    try:
        await pool.set(key, json.dumps({"worker_id": worker_id}), ex=WORKER_HEARTBEAT_TTL)
    except Exception:
        logger.exception("heartbeat_write_failed", worker_id=worker_id)


async def get_active_workers() -> list[str]:
    """Return the IDs of workers whose heartbeats are still fresh."""
    pool = await get_redis_pool()
    try:
        keys = await pool.keys(f"{WORKER_HEARTBEAT_PREFIX}*")
        return [k.split(":", 2)[2] for k in keys] if keys else []
    except Exception:
        logger.exception("heartbeat_scan_failed")
        return []


async def clear_worker_heartbeat(worker_id: str) -> None:
    """Remove a worker's heartbeat key (used during graceful shutdown)."""
    pool = await get_redis_pool()
    key = worker_heartbeat_key(worker_id)
    try:
        await pool.delete(key)
    except Exception:
        logger.exception("heartbeat_clear_failed", worker_id=worker_id)


# ── Task enqueue helpers ────────────────────────────────────────────

# Task type → arq function name mapping
TASK_FUNCTION_MAP: dict[str, str] = {
    "document.ingest": "document_ingest",
    "document.ingest_text": "document_ingest_text",
    "document.index": "document_index",
    "vs_file.process": "vs_file_process",
}


async def enqueue_task(
    task_type: str,
    payload: dict[str, Any],
    *,
    queue: str = "default",
) -> str | None:
    """Enqueue a background task via arq.

    Parameters
    ----------
    task_type:
        One of ``document.ingest``, ``document.ingest_text``,
        ``document.index``, ``vs_file.process``.
    payload:
        Task-specific arguments forwarded to the worker function.
    queue:
        arq queue name (default ``"default"``).

    Returns
    -------
    The arq job key, or ``None`` if enqueue failed.
    """
    fn_name = TASK_FUNCTION_MAP.get(task_type)
    if fn_name is None:
        logger.error("unknown_task_type", task_type=task_type)
        return None

    pool = await get_redis_pool()
    try:
        job = await pool.enqueue_job(
            fn_name,
            _queue_name=queue,
            _job_id=payload.get("document_id") or payload.get("vector_store_file_id"),
            **payload,
        )
        if job is not None:
            logger.info(
                "task_enqueued",
                task_type=task_type,
                job_id=job.job_id,
                queue=queue,
            )
            return job.job_id
        else:
            # arq returns None when a job with the same _job_id already exists
            # (deduplication). This is fine — the existing job is still running.
            logger.info(
                "task_deduplicated",
                task_type=task_type,
                job_id=payload.get("document_id") or payload.get("vector_store_file_id"),
            )
            return None
    except Exception as exc:
        logger.error(
            "task_enqueue_failed",
            task_type=task_type,
            error=str(exc),
            error_type=type(exc).__name__,
            exc_info=True,
        )
        return None
