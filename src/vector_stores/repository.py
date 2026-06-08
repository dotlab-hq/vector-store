"""Repository layer for vector_stores and vector_store_files tables."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import (
    DocumentModel,
    VectorStoreFileBatchModel,
    VectorStoreFileModel,
    VectorStoreModel,
)


def _utcnow() -> datetime:
    """Return a timezone-naive UTC datetime — matches SQLAlchemy's func.now()."""
    import datetime as _dt

    return _dt.datetime.utcnow()


class VectorStoreRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    # ── CRUD ──────────────────────────────────────────────────────────

    async def create(self, store: VectorStoreModel) -> VectorStoreModel:
        self.session.add(store)
        await self.session.flush()
        return store

    async def get(self, store_id: str) -> VectorStoreModel | None:
        return await self.session.get(VectorStoreModel, store_id)

    async def list_all(
        self,
        *,
        limit: int = 20,
        after_id: str | None = None,
    ) -> list[VectorStoreModel]:
        stmt = select(VectorStoreModel).order_by(VectorStoreModel.created_at.desc())
        if after_id:
            # "after" means created before this id
            after_store = await self.get(after_id)
            if after_store:
                stmt = stmt.where(VectorStoreModel.created_at < after_store.created_at)
        stmt = stmt.limit(limit + 1)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def update(
        self,
        store_id: str,
        *,
        name: str | None = None,
        metadata_json: str | None = None,
        expires_at: datetime | None = None,
        expires_after_days: int | None = None,
        status: str | None = None,
        last_active_at: datetime | None = None,
    ) -> VectorStoreModel | None:
        values: dict[str, object] = {}
        if name is not None:
            values["name"] = name
        if metadata_json is not None:
            values["metadata_json"] = metadata_json
        if expires_at is not None:
            values["expires_at"] = expires_at
        if expires_after_days is not None:
            values["expires_after_days"] = expires_after_days
        if status is not None:
            values["status"] = status
        if last_active_at is not None:
            values["last_active_at"] = last_active_at
        if not values:
            return await self.get(store_id)
        await self.session.execute(
            update(VectorStoreModel)
            .where(VectorStoreModel.id == store_id)
            .values(**values)
        )
        await self.session.flush()
        return await self.get(store_id)

    async def delete(self, store_id: str) -> bool:
        store = await self.get(store_id)
        if store is None:
            return False
        await self.session.delete(store)
        await self.session.flush()
        return True


class VectorStoreFileRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    # ── CRUD ──────────────────────────────────────────────────────────

    async def create(self, vf: VectorStoreFileModel) -> VectorStoreFileModel:
        self.session.add(vf)
        await self.session.flush()
        return vf

    async def get(self, file_id: str) -> VectorStoreFileModel | None:
        return await self.session.get(VectorStoreFileModel, file_id)

    async def get_by_document_id(self, document_id: str) -> list[VectorStoreFileModel]:
        """Return all vector-store file rows linked to a source document."""
        result = await self.session.execute(
            select(VectorStoreFileModel).where(
                VectorStoreFileModel.source_document_id == document_id
            )
        )
        return list(result.scalars().all())

    async def get_by_store_and_document(
        self, store_id: str, document_id: str
    ) -> VectorStoreFileModel | None:
        result = await self.session.execute(
            select(VectorStoreFileModel).where(
                VectorStoreFileModel.vector_store_id == store_id,
                VectorStoreFileModel.source_document_id == document_id,
            )
        )
        return result.scalars().first()

    async def list_by_store(
        self,
        store_id: str,
        *,
        limit: int = 20,
        after_id: str | None = None,
        before_id: str | None = None,
        status_filter: str | None = None,
        order: str = "desc",
    ) -> list[VectorStoreFileModel]:
        stmt = select(VectorStoreFileModel).where(
            VectorStoreFileModel.vector_store_id == store_id
        )

        # Status filter
        if status_filter:
            stmt = stmt.where(VectorStoreFileModel.status == status_filter)

        # Ordering
        sort_col = VectorStoreFileModel.created_at
        if order == "asc":
            stmt = stmt.order_by(sort_col.asc())
        else:
            stmt = stmt.order_by(sort_col.desc())

        # Cursor pagination: after
        if after_id:
            after_file = await self.get(after_id)
            if after_file:
                if order == "asc":
                    stmt = stmt.where(
                        VectorStoreFileModel.created_at > after_file.created_at
                    )
                else:
                    stmt = stmt.where(
                        VectorStoreFileModel.created_at < after_file.created_at
                    )

        # Cursor pagination: before
        if before_id:
            before_file = await self.get(before_id)
            if before_file:
                if order == "asc":
                    stmt = stmt.where(
                        VectorStoreFileModel.created_at < before_file.created_at
                    )
                else:
                    stmt = stmt.where(
                        VectorStoreFileModel.created_at > before_file.created_at
                    )

        stmt = stmt.limit(limit + 1)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def update_status(
        self,
        file_id: str,
        *,
        status: str,
        failure_reason: str | None = None,
        next_attempt_at: datetime | None = None,
        locked_at: datetime | None = None,
        locked_by: str | None = None,
        attempts: int | None = None,
        completed_at: datetime | None = None,
    ) -> VectorStoreFileModel | None:
        values: dict[str, object] = {"status": status}
        if failure_reason is not None:
            values["failure_reason"] = failure_reason
        if next_attempt_at is not None:
            values["next_attempt_at"] = next_attempt_at
        if locked_at is not None:
            values["locked_at"] = locked_at
        if locked_by is not None:
            values["locked_by"] = locked_by
        if attempts is not None:
            values["attempts"] = attempts
        if completed_at is not None:
            values["completed_at"] = completed_at
        await self.session.execute(
            update(VectorStoreFileModel)
            .where(VectorStoreFileModel.id == file_id)
            .values(**values)
        )
        await self.session.flush()
        return await self.get(file_id)

    async def cancel_by_document_id(self, document_id: str) -> int:
        """Mark all vector-store rows for a document as cancelled."""
        result = await self.session.execute(
            update(VectorStoreFileModel)
            .where(VectorStoreFileModel.source_document_id == document_id)
            .where(VectorStoreFileModel.status != "cancelled")
            .values(
                status="cancelled",
                locked_at=None,
                locked_by=None,
                next_attempt_at=None,
                failure_reason=None,
            )
        )
        await self.session.flush()
        return int(result.rowcount or 0)

    async def delete_by_document_id(self, document_id: str) -> int:
        """Delete all vector-store file rows for a document."""
        result = await self.session.execute(
            delete(VectorStoreFileModel).where(
                VectorStoreFileModel.source_document_id == document_id
            )
        )
        return int(result.rowcount or 0)

    async def delete_by_file_id(self, file_id: str) -> bool:
        """Delete a single vector-store file row by its ID."""
        result = await self.session.execute(
            delete(VectorStoreFileModel).where(VectorStoreFileModel.id == file_id)
        )
        return (result.rowcount or 0) > 0

    async def claim_pending(
        self,
        *,
        limit: int = 2,
        lease_minutes: int = 10,
        worker_id: str = "worker-0",
        max_retries: int = 5,
        retry_cap_s: int = 3600,
    ) -> list[VectorStoreFileModel]:
        """Claim up to ``limit`` files ready for processing using SKIP LOCKED.

        Returns claimed files with status set to 'processing' and lease acquired.
        Skips files whose batch has been cancelled.
        """
        now = _utcnow()
        lease_cutoff = now - timedelta(minutes=lease_minutes)

        # Select eligible rows. Exclude files whose batch is cancelled.
        from src.database.models import VectorStoreFileBatchModel

        stmt = (
            select(VectorStoreFileModel)
            .outerjoin(
                VectorStoreFileBatchModel,
                VectorStoreFileModel.batch_id == VectorStoreFileBatchModel.id,
            )
            .where(
                (
                    (VectorStoreFileModel.status == "pending")
                    | (
                        (VectorStoreFileModel.status == "failed")
                        & (VectorStoreFileModel.next_attempt_at <= now)
                        & (VectorStoreFileModel.attempts < max_retries)
                    )
                )
                & (
                    (VectorStoreFileModel.locked_at.is_(None))
                    | (VectorStoreFileModel.locked_at < lease_cutoff)
                )
                & (
                    (VectorStoreFileModel.batch_id.is_(None))
                    | (VectorStoreFileBatchModel.status != "cancelled")
                )
            )
            .order_by(VectorStoreFileModel.created_at)
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        result = await self.session.execute(stmt)
        files = list(result.scalars().all())

        if not files:
            return []

        # Update each claimed row
        claim_updates = []
        for f in files:
            new_attempts = f.attempts + 1
            claim_updates.append(
                {
                    "id": f.id,
                    "status": "processing",
                    "locked_at": now,
                    "locked_by": worker_id,
                    "attempts": new_attempts,
                }
            )

        for update_data in claim_updates:
            file_id = update_data.pop("id")
            await self.session.execute(
                update(VectorStoreFileModel)
                .where(VectorStoreFileModel.id == file_id)
                .values(**update_data)
            )
        await self.session.flush()

        # Return refreshed copies
        refreshed = []
        for ud in claim_updates:
            refreshed.append(await self.get(ud["id"]))
        return refreshed

    async def mark_failed(
        self,
        file_id: str,
        *,
        failure_reason: str,
        next_attempt_at: datetime | None = None,
    ) -> None:
        await self.update_status(
            file_id,
            status="failed",
            failure_reason=failure_reason,
            next_attempt_at=next_attempt_at,
            locked_at=None,
            locked_by=None,
        )

    async def mark_completed(self, file_id: str) -> None:
        await self.update_status(
            file_id,
            status="completed",
            completed_at=_utcnow(),
            locked_at=None,
            locked_by=None,
            failure_reason=None,
        )

    async def complete_pending_for_document(self, document_id: str) -> list[str]:
        """Mark all non-terminal VF rows for a document as completed.

        Returns the list of affected vector_store_ids so callers can
        recompute file_counts for each store.
        """
        result = await self.session.execute(
            select(VectorStoreFileModel.vector_store_id).where(
                VectorStoreFileModel.source_document_id == document_id,
                VectorStoreFileModel.status.notin_(
                    ["completed", "cancelled", "failed"]
                ),
            )
        )
        affected_store_ids = list({row[0] for row in result.fetchall()})

        await self.session.execute(
            update(VectorStoreFileModel)
            .where(
                VectorStoreFileModel.source_document_id == document_id,
                VectorStoreFileModel.status.notin_(
                    ["completed", "cancelled", "failed"]
                ),
            )
            .values(
                status="completed",
                completed_at=_utcnow(),
                failure_reason=None,
                locked_at=None,
                locked_by=None,
            )
        )
        await self.session.flush()
        return affected_store_ids

    async def update_attributes(
        self, file_id: str, attributes: dict
    ) -> VectorStoreFileModel | None:
        attributes_json = json.dumps(attributes)
        await self.session.execute(
            update(VectorStoreFileModel)
            .where(VectorStoreFileModel.id == file_id)
            .values(attributes_json=attributes_json)
        )
        await self.session.flush()
        return await self.get(file_id)

    async def update_usage_bytes(self, file_id: str, usage_bytes: int) -> None:
        await self.session.execute(
            update(VectorStoreFileModel)
            .where(VectorStoreFileModel.id == file_id)
            .values(bytes=usage_bytes)
        )
        await self.session.flush()

    async def sweep_failed_for_retry(self, *, max_retries: int = 5) -> int:
        """Re-promote eligible failed rows back to pending.

        Returns number of rows promoted.
        """
        now = _utcnow()
        result = await self.session.execute(
            update(VectorStoreFileModel)
            .where(
                VectorStoreFileModel.status == "failed",
                VectorStoreFileModel.next_attempt_at <= now,
                VectorStoreFileModel.attempts < max_retries,
            )
            .values(
                status="pending",
                next_attempt_at=None,
                locked_at=None,
                locked_by=None,
            )
        )
        await self.session.flush()
        return int(result.rowcount or 0)

    async def release_stale_processing(
        self, *, stale_minutes: int = 10, max_retries: int = 5
    ) -> int:
        """Reset files stuck in processing/chunking/embedding back to failed.

        These occur when the worker crashes mid-processing and never marks
        the file as failed or completed. The locked_at timestamp is used
        to determine staleness.
        """
        cutoff = _utcnow() - timedelta(minutes=stale_minutes)
        result = await self.session.execute(
            update(VectorStoreFileModel)
            .where(
                VectorStoreFileModel.status.in_(
                    ["processing", "chunking", "embedding", "indexing"]
                ),
                VectorStoreFileModel.locked_at.is_not(None),
                VectorStoreFileModel.locked_at < cutoff,
                VectorStoreFileModel.attempts < max_retries,
            )
            .values(
                status="failed",
                failure_reason="worker_lost_lock",
                next_attempt_at=_utcnow(),
                locked_at=None,
                locked_by=None,
            )
        )
        await self.session.flush()
        return int(result.rowcount or 0)

    async def mark_permanently_failed(self, *, max_retries: int = 5) -> int:
        """Mark rows that have exhausted all retries."""
        result = await self.session.execute(
            update(VectorStoreFileModel)
            .where(
                VectorStoreFileModel.status == "failed",
                VectorStoreFileModel.attempts >= max_retries,
                VectorStoreFileModel.failure_reason.is_(None),
            )
            .values(failure_reason="max_retries_exceeded")
        )
        await self.session.flush()
        return int(result.rowcount or 0)

    async def update_store_file_counts(self, store_id: str) -> dict[str, int]:
        """Recompute and write file_counts_json on the parent store."""
        result = await self.session.execute(
            select(VectorStoreFileModel.status).where(
                VectorStoreFileModel.vector_store_id == store_id
            )
        )
        statuses = [row[0] for row in result.fetchall()]
        counts = {
            "in_progress": 0,
            "completed": 0,
            "cancelled": 0,
            "failed": 0,
            "total": len(statuses),
        }
        for s in statuses:
            if s == "completed":
                counts["completed"] += 1
            elif s == "cancelled":
                counts["cancelled"] += 1
            elif s == "failed":
                counts["failed"] += 1
            elif s not in ("completed", "cancelled", "failed"):
                counts["in_progress"] += 1

        counts_json = json.dumps(counts)
        # Transition store status based on file states:
        #   in_progress if any files are still being processed,
        #   completed   if all files are terminal.
        store_status = "in_progress" if counts["in_progress"] > 0 else "completed"
        await self.session.execute(
            update(VectorStoreModel)
            .where(VectorStoreModel.id == store_id)
            .values(file_counts_json=counts_json, status=store_status)
        )
        await self.session.flush()
        return counts

    async def get_documents_by_store(self, store_id: str) -> dict[str, DocumentModel]:
        result = await self.session.execute(
            select(DocumentModel)
            .join(
                VectorStoreFileModel,
                VectorStoreFileModel.source_document_id == DocumentModel.id,
            )
            .where(VectorStoreFileModel.vector_store_id == store_id)
        )
        return {doc.id: doc for doc in result.scalars().all()}


class VectorStoreFileBatchRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self, batch: VectorStoreFileBatchModel
    ) -> VectorStoreFileBatchModel:
        self.session.add(batch)
        await self.session.flush()
        return batch

    async def get(self, batch_id: str) -> VectorStoreFileBatchModel | None:
        return await self.session.get(VectorStoreFileBatchModel, batch_id)

    async def list_files_in_batch(
        self,
        batch_id: str,
        *,
        limit: int = 20,
        after_id: str | None = None,
        before_id: str | None = None,
        status_filter: str | None = None,
        order: str = "desc",
    ) -> list[VectorStoreFileModel]:
        stmt = select(VectorStoreFileModel).where(
            VectorStoreFileModel.batch_id == batch_id
        )
        if status_filter:
            stmt = stmt.where(VectorStoreFileModel.status == status_filter)

        sort_col = VectorStoreFileModel.created_at
        if order == "asc":
            stmt = stmt.order_by(sort_col.asc())
        else:
            stmt = stmt.order_by(sort_col.desc())

        if after_id:
            after_file = await self.session.get(VectorStoreFileModel, after_id)
            if after_file:
                if order == "asc":
                    stmt = stmt.where(
                        VectorStoreFileModel.created_at > after_file.created_at
                    )
                else:
                    stmt = stmt.where(
                        VectorStoreFileModel.created_at < after_file.created_at
                    )

        if before_id:
            before_file = await self.session.get(VectorStoreFileModel, before_id)
            if before_file:
                if order == "asc":
                    stmt = stmt.where(
                        VectorStoreFileModel.created_at < before_file.created_at
                    )
                else:
                    stmt = stmt.where(
                        VectorStoreFileModel.created_at > before_file.created_at
                    )

        stmt = stmt.limit(limit + 1)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def cancel(self, batch_id: str) -> VectorStoreFileBatchModel | None:
        """Mark batch cancelled. Worker will skip its pending files going forward."""
        result = await self.session.execute(
            update(VectorStoreFileBatchModel)
            .where(VectorStoreFileBatchModel.id == batch_id)
            .where(VectorStoreFileBatchModel.status != "cancelled")
            .values(status="cancelled", cancelled_at=_utcnow())
        )
        if (result.rowcount or 0) == 0:
            return await self.get(batch_id)
        await self.session.flush()
        return await self.get(batch_id)

    async def update_file_counts(self, batch_id: str) -> dict[str, int]:
        """Recompute and persist file_counts_json for a batch."""
        result = await self.session.execute(
            select(VectorStoreFileModel.status).where(
                VectorStoreFileModel.batch_id == batch_id
            )
        )
        statuses = [row[0] for row in result.fetchall()]
        counts = {
            "in_progress": 0,
            "completed": 0,
            "cancelled": 0,
            "failed": 0,
            "total": len(statuses),
        }
        for s in statuses:
            if s == "completed":
                counts["completed"] += 1
            elif s == "cancelled":
                counts["cancelled"] += 1
            elif s == "failed":
                counts["failed"] += 1
            elif s not in ("completed", "cancelled", "failed"):
                counts["in_progress"] += 1

        counts_json = json.dumps(counts)
        await self.session.execute(
            update(VectorStoreFileBatchModel)
            .where(VectorStoreFileBatchModel.id == batch_id)
            .values(file_counts_json=counts_json)
        )
        await self.session.flush()
        return counts

    async def derive_status(self, batch_id: str) -> str:
        """Roll up batch status from file statuses.

        - all cancelled or terminal-with-no-active -> cancelled
        - all completed -> completed
        - any failed and no in_progress -> failed
        - otherwise -> in_progress
        """
        result = await self.session.execute(
            select(VectorStoreFileModel.status).where(
                VectorStoreFileModel.batch_id == batch_id
            )
        )
        statuses = [row[0] for row in result.fetchall()]
        if not statuses:
            return "completed"
        terminal = {"completed", "cancelled", "failed"}
        if all(s == "cancelled" for s in statuses):
            return "cancelled"
        if all(s in terminal for s in statuses):
            # Mixed terminal: prefer "completed" if any completed, else "cancelled" or "failed"
            if any(s == "completed" for s in statuses) and not any(
                s == "failed" for s in statuses
            ):
                return "completed"
            if any(s == "failed" for s in statuses) and not any(
                s == "completed" for s in statuses
            ):
                return "failed"
            return "completed"
        return "in_progress"

    async def mark_completed(self, batch_id: str) -> None:
        await self.session.execute(
            update(VectorStoreFileBatchModel)
            .where(VectorStoreFileBatchModel.id == batch_id)
            .values(status="completed", completed_at=_utcnow())
        )
        await self.session.flush()
