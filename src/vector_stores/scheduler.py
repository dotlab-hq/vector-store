"""Scheduler — owns cron tasks; provides start() / stop() for lifespan.

The VectorStoreWorker has been replaced by arq (see ``apps.worker.arq_settings``).
Only the retry cron remains here.
"""

from __future__ import annotations

from src.config import settings
from src.observability.logging import get_logger
from src.vector_stores.cron import VectorStoreCron

logger = get_logger()


class VectorStoreScheduler:
    """Manages the background cron task."""

    def __init__(self) -> None:
        self.cron = VectorStoreCron()

    def start(self) -> None:
        self.cron.start()
        logger.info("vs_scheduler_started")

    async def stop(self) -> None:
        await self.cron.stop()
        logger.info("vs_scheduler_stopped")
