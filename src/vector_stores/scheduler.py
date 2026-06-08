"""Scheduler — owns worker + cron tasks; provides start() / stop() for lifespan."""

from __future__ import annotations

from src.config import settings
from src.observability.logging import get_logger
from src.vector_stores.cron import VectorStoreCron
from src.vector_stores.worker import VectorStoreWorker

logger = get_logger()


class VectorStoreScheduler:
    """Manages the background worker and cron tasks."""

    def __init__(self) -> None:
        self.worker = VectorStoreWorker()
        self.cron = VectorStoreCron()

    def start(self) -> None:
        if not settings.vector_store_worker_enabled:
            logger.info("vs_scheduler_disabled")
            return
        self.worker.start()
        self.cron.start()
        logger.info("vs_scheduler_started")

    async def stop(self) -> None:
        await self.worker.stop()
        await self.cron.stop()
        logger.info("vs_scheduler_stopped")
