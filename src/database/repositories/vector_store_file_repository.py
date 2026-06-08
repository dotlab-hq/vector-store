from datetime import UTC, datetime

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import VectorStoreFileBatchModel, VectorStoreFileModel, VectorStoreModel


class VectorStoreFileRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def complete_pending_for_document(self, document_id: str) -> list[str]:
        """Mark all *pending* vector_store_files for a document as completed.

        Returns the distinct ``vector_store_id`` values that were touched so the
        caller can refresh counts / tags on those stores.
        """
        now = datetime.now(UTC)

        # Fetch pending files to discover affected stores
        pending = await self.session.execute(
            select(VectorStoreFileModel.vector_store_id).where(
                VectorStoreFileModel.source_document_id == document_id,
                VectorStoreFileModel.status == "pending",
            ).distinct()
        )
        store_ids: list[str] = [row[0] for row in pending.all()]
        if not store_ids:
            return []

        await self.session.execute(
            update(VectorStoreFileModel)
            .where(
                VectorStoreFileModel.source_document_id == document_id,
                VectorStoreFileModel.status == "pending",
            )
            .values(status="completed", completed_at=now)
        )
        await self.session.flush()

        return store_ids

    async def update_store_file_counts(self, vector_store_id: str) -> None:
        """Recompute ``file_counts_json`` and ``usage_bytes`` for a vector store."""
        row = await self.session.execute(
            select(
                VectorStoreFileModel.status,
                func.count(VectorStoreFileModel.id),
                func.coalesce(func.sum(VectorStoreFileModel.bytes), 0),
            )
            .where(VectorStoreFileModel.vector_store_id == vector_store_id)
            .group_by(VectorStoreFileModel.status)
        )
        counts = {"in_progress": 0, "completed": 0, "cancelled": 0, "failed": 0, "total": 0}
        total_bytes = 0
        for status, cnt, bts in row.all():
            counts[status] = cnt
            total_bytes += bts
            counts["total"] += cnt

        import json

        await self.session.execute(
            update(VectorStoreModel)
            .where(VectorStoreModel.id == vector_store_id)
            .values(
                file_counts_json=json.dumps(counts),
                usage_bytes=total_bytes,
            )
        )
        await self.session.flush()
