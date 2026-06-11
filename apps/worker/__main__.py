"""Standalone worker entrypoint using arq.

Run as: ``python -m apps.worker``
  — or —
Run as: ``arq apps.worker.arq_settings.WorkerSettings``

The ``__main__`` block starts an arq Worker that polls Redis for jobs
published by the API and dispatches them to the handler functions in
``apps.worker.arq_settings``.
"""

from __future__ import annotations

import asyncio
import signal
from typing import Any
from uuid import uuid4

from arq import Worker
from arq.worker import WorkerStatus

from src.config import settings
from src.database import engine
from src.database.repositories import DocumentRepository
from src.database.session import async_session_factory
from src.indexing.bm25.bm25_store import Bm25Store
from src.observability.logging import get_logger, setup_logging
from src.shared.events import (
    clear_worker_heartbeat,
    get_active_workers,
    get_redis_settings,
    update_worker_heartbeat,
)
from apps.worker.arq_settings import WorkerSettings

setup_logging()
logger = get_logger()


async def rebuild_bm25_on_startup() -> None:
    """Rebuild BM25 in-memory index before the worker starts accepting jobs."""
    bm25 = Bm25Store()
    for attempt in range(1, 4):
        try:
            count = await bm25.rebuild_from_db()
            if count:
                logger.info("bm25_rebuilt", count=count)
            return
        except OSError as exc:
            logger.warning("bm25_rebuild_retry", attempt=attempt, error=str(exc))
            await asyncio.sleep(2 * attempt)
    logger.error("bm25_rebuild_failed", error="exhausted retries")


async def _release_stale_documents() -> None:
    """Reset documents stuck in ``processing`` status from a previous crash."""
    async with async_session_factory() as session:
        repo = DocumentRepository(session)
        stale_ids = await repo.release_stale_documents(
            stale_minutes=settings.task_worker_lease_minutes
        )
        if stale_ids:
            logger.info(
                "startup_released_stale_documents",
                count=len(stale_ids),
                ids=stale_ids[:20],
            )


async def _release_stale_tasks() -> None:
    """Release stale ProcessingTaskModel entries from a previous crash."""
    from src.database.repositories import ProcessingTaskRepository

    async with async_session_factory() as session:
        task_repo = ProcessingTaskRepository(session)
        released = await task_repo.release_stale(
            stale_minutes=settings.task_worker_lease_minutes
        )
        await session.commit()
        if released:
            logger.info("startup_released_orphaned_tasks", count=released)


async def _heartbeat_loop(worker_id: str) -> None:
    """Periodically write a heartbeat key so the cron can detect live workers."""
    while True:
        try:
            await update_worker_heartbeat(worker_id)
            active = await get_active_workers()
            logger.debug("heartbeat_tick", worker_id=worker_id, active_count=len(active))
        except Exception:
            logger.exception("heartbeat_tick_failed", worker_id=worker_id)
        await asyncio.sleep(15)


async def main() -> None:
    worker_id = f"worker-{uuid4().hex[:8]}"

    # Release stale state from a previous crash
    await _release_stale_tasks()
    await _release_stale_documents()

    # Rebuild BM25 index
    try:
        await rebuild_bm25_on_startup()
    except Exception as e:
        logger.warning("bm25_startup_failed", error=str(e))

    # Start the arq worker
    redis_settings = get_redis_settings()
    worker = Worker(
        WorkerSettings.functions,
        redis_settings=redis_settings,
        queue_name=WorkerSettings.queue_name,
        max_jobs=WorkerSettings.max_jobs,
        poll_delay=WorkerSettings.poll_delay,
        max_tries=WorkerSettings.max_tries,
        health_check_interval=WorkerSettings.health_check_interval,
        on_startup=None,
        on_shutdown=None,
    )

    stop_event = asyncio.Event()

    def _on_signal(signum: int, _frame: Any) -> None:
        logger.info("worker_signal_received", signum=signum)
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _on_signal, sig, None)
        except NotImplementedError:
            signal.signal(sig, _on_signal)

    logger.info(
        "arq_worker_started",
        worker_id=worker_id,
        max_jobs=WorkerSettings.max_jobs,
        queue=WorkerSettings.queue_name,
    )

    # Start heartbeat background task
    heartbeat_task = asyncio.create_task(
        _heartbeat_loop(worker_id), name="worker-heartbeat"
    )

    # arq Worker.run() blocks until cancelled — run it in a task
    worker_task = asyncio.create_task(worker.run(), name="arq-worker")

    try:
        await stop_event.wait()
    finally:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
        await clear_worker_heartbeat(worker_id)

        await worker.close()
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
        await engine.dispose()
        logger.info("worker_shutdown_complete")


if __name__ == "__main__":
    asyncio.run(main())
