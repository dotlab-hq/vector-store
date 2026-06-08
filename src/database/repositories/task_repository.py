"""Repository for the processing_tasks table (PostgreSQL-backed job queue)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from uuid import uuid4

from sqlalchemy import case, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import ProcessingTaskModel


def _utcnow() -> datetime:
    return datetime.utcnow()


class ProcessingTaskRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        task_type: str,
        payload: dict,
        *,
        priority: int = 0,
        max_retries: int = 5,
    ) -> ProcessingTaskModel:
        task = ProcessingTaskModel(
            id=uuid4().hex,
            task_type=task_type,
            payload_json=json.dumps(payload),
            status="pending",
            priority=priority,
            max_retries=max_retries,
        )
        self.session.add(task)
        await self.session.flush()
        return task

    async def get(self, task_id: str) -> ProcessingTaskModel | None:
        return await self.session.get(ProcessingTaskModel, task_id)

    async def claim_pending(
        self,
        *,
        limit: int = 5,
        lease_minutes: int = 15,
        worker_id: str = "worker-0",
    ) -> list[ProcessingTaskModel]:
        """Claim up to *limit* tasks using SELECT ... FOR UPDATE SKIP LOCKED."""
        now = _utcnow()
        lease_cutoff = now - timedelta(minutes=lease_minutes)

        stmt = (
            select(ProcessingTaskModel)
            .where(
                (
                    (ProcessingTaskModel.status == "pending")
                    | (
                        (ProcessingTaskModel.status == "failed")
                        & (ProcessingTaskModel.next_attempt_at <= now)
                        & (ProcessingTaskModel.attempts < ProcessingTaskModel.max_retries)
                    )
                )
                & (
                    (ProcessingTaskModel.locked_at.is_(None))
                    | (ProcessingTaskModel.locked_at < lease_cutoff)
                )
            )
            .order_by(
                ProcessingTaskModel.priority.desc(), ProcessingTaskModel.created_at
            )
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        result = await self.session.execute(stmt)
        tasks = list(result.scalars().all())

        if not tasks:
            return []

        claim_time = _utcnow()

        # Bulk UPDATE using a CASE expression to set per-task attempts
        task_ids = [t.id for t in tasks]
        await self.session.execute(
            update(ProcessingTaskModel)
            .where(ProcessingTaskModel.id.in_(task_ids))
            .values(
                status="processing",
                locked_at=claim_time,
                locked_by=worker_id,
                attempts=case(
                    *[
                        (ProcessingTaskModel.id == t.id, t.attempts + 1)
                        for t in tasks
                    ],
                    else_=ProcessingTaskModel.attempts,
                ),
            )
        )
        await self.session.flush()

        # Re-fetch updated rows in a single SELECT
        result = await self.session.execute(
            select(ProcessingTaskModel).where(
                ProcessingTaskModel.id.in_(task_ids)
            )
        )
        return list(result.scalars().all())

    async def mark_completed(self, task_id: str) -> None:
        await self.session.execute(
            update(ProcessingTaskModel)
            .where(ProcessingTaskModel.id == task_id)
            .values(
                status="completed",
                locked_at=None,
                locked_by=None,
                last_error=None,
            )
        )
        await self.session.flush()

    async def mark_failed(
        self,
        task_id: str,
        *,
        error: str,
        next_attempt_at: datetime | None = None,
        max_retries: int = 5,
        attempts: int = 0,
    ) -> None:
        if attempts >= max_retries:
            await self.session.execute(
                update(ProcessingTaskModel)
                .where(ProcessingTaskModel.id == task_id)
                .values(
                    status="permanently_failed",
                    last_error=error,
                    locked_at=None,
                    locked_by=None,
                )
            )
        else:
            await self.session.execute(
                update(ProcessingTaskModel)
                .where(ProcessingTaskModel.id == task_id)
                .values(
                    status="failed",
                    last_error=error,
                    next_attempt_at=next_attempt_at,
                    locked_at=None,
                    locked_by=None,
                )
            )
        await self.session.flush()

    async def release_stale(self, *, stale_minutes: int = 15) -> int:
        """Reset tasks stuck in 'processing' past their lease."""
        cutoff = _utcnow() - timedelta(minutes=stale_minutes)
        result = await self.session.execute(
            update(ProcessingTaskModel)
            .where(
                ProcessingTaskModel.status == "processing",
                ProcessingTaskModel.locked_at.is_not(None),
                ProcessingTaskModel.locked_at < cutoff,
            )
            .values(
                status="pending",
                locked_at=None,
                locked_by=None,
            )
        )
        await self.session.flush()
        return int(result.rowcount or 0)
