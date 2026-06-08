"""Standalone worker entrypoint.

Run as: ``python -m apps.worker``.
"""

import asyncio
import signal
from typing import Any
from uuid import uuid4

from src.database import engine
from src.database.session import async_session_factory
from src.config import settings
from src.observability.logging import get_logger, setup_logging
from apps.worker.processor import TaskProcessor
from src.vector_stores.cron import VectorStoreCron

setup_logging()
logger = get_logger()


async def main() -> None:
    worker_id = f"worker-{uuid4().hex[:8]}"
    processor = TaskProcessor(worker_id)
    processor.init_deps()

    # Recover orphaned tasks from a previous crash before starting the loop
    from src.database.repositories import ProcessingTaskRepository

    async with async_session_factory() as session:
        task_repo = ProcessingTaskRepository(session)
        released = await task_repo.release_stale(
            stale_minutes=settings.task_worker_lease_minutes
        )
        await session.commit()
        if released:
            logger.info("startup_released_orphaned_tasks", count=released)

    # Rebuild BM25 in-memory index from the database so the index is not empty
    try:
        await processor.rebuild_bm25()
    except Exception as e:
        logger.warning("bm25_rebuild_failed", error=str(e))

    cron = VectorStoreCron()
    if settings.vector_store_worker_enabled:
        cron.start()

    processor.start()

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

    try:
        await stop_event.wait()
    finally:
        await processor.stop()
        await cron.stop()
        await engine.dispose()
        logger.info("worker_shutdown_complete")


if __name__ == "__main__":
    asyncio.run(main())
