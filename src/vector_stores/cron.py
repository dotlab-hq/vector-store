"""Retry cron - periodically re-promotes failed files and expires stores.

Runs as an asyncio task alongside the worker.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from sqlalchemy import update

from src.config import settings
from src.database.models import VectorStoreModel
from src.database.session import async_session_factory
from src.observability.logging import get_logger
from src.vector_stores.repository import VectorStoreFileRepository

logger = get_logger()


def _utcnow() -> datetime:
    return datetime.utcnow()


class VectorStoreCron:
    """Periodic sweep: retry failed files, expire stores, mark permanent failures."""

    def __init__(self) -> None:
        self._shutdown = False
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run_loop(), name="vs-cron")
        logger.info("vs_cron_started")

    async def stop(self) -> None:
        self._shutdown = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("vs_cron_stopped")

    async def _run_loop(self) -> None:
        interval = settings.vector_store_cron_interval_s
        while not self._shutdown:
            try:
                await self._sweep()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("vs_cron_sweep_error")
            await asyncio.sleep(interval)

    async def _sweep(self) -> None:
        async with async_session_factory() as session:
            vf_repo = VectorStoreFileRepository(session)
            max_retries = settings.vector_store_retry_max
            lease_minutes = settings.vector_store_worker_lease_minutes

            # 0. Recover files stuck in intermediate statuses (processing, chunking, etc.)
            #    This happens when the worker crashes before marking failed.
            orphaned = await vf_repo.release_stale_processing(
                stale_minutes=lease_minutes,
                max_retries=max_retries,
            )
            if orphaned:
                logger.info("vs_cron_released_stale_files", count=orphaned)

            # 1. Re-promote eligible failed rows back to pending
            promoted = await vf_repo.sweep_failed_for_retry(max_retries=max_retries)
            if promoted:
                logger.info("vs_cron_promoted", count=promoted)

            # 2. Mark permanently failed rows
            marked = await vf_repo.mark_permanently_failed(max_retries=max_retries)
            if marked:
                logger.info("vs_cron_marked_permanently_failed", count=marked)

            # 3. Expire vector stores whose expires_at has passed
            now = _utcnow()
            result = await session.execute(
                update(VectorStoreModel)
                .where(
                    VectorStoreModel.status != "expired",
                    VectorStoreModel.expires_at.is_not(None),
                    VectorStoreModel.expires_at <= now,
                )
                .values(status="expired")
            )
            expired_count = int(result.rowcount or 0)
            if expired_count:
                logger.info("vs_cron_expired_stores", count=expired_count)

            await session.commit()
