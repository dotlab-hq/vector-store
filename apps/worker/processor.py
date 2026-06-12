"""Background task processor — polls processing_tasks and handles all heavy work.

Replaces the in-process BackgroundTasks used by the API and the standalone
VectorStoreWorker.  Runs as a separate process (``python -m apps.worker``).
"""

from __future__ import annotations

import asyncio
import json
import random
import tempfile
import traceback
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

from sqlalchemy import update as sa_update

from src.config import settings
from src.database.models import (
    ProcessingTaskModel,
    VectorStoreFileModel,
)
from src.database.repositories import DocumentRepository, ProcessingTaskRepository
from src.database.repositories.task_repository import _utcnow
from src.database.session import async_session_factory
from src.indexing.bm25.bm25_store import Bm25Store
from src.indexing.embeddings import embeddings
from src.indexing.indexer import Indexer
from src.indexing.qdrant.qdrant_store import QdrantVectorStore
from src.observability.logging import get_logger

logger = get_logger()


def _backoff(attempt: int) -> float:
    """Exponential backoff with jitter."""
    base = settings.task_worker_retry_base_s
    cap = settings.task_worker_retry_cap_s
    delay = min(base * (2 ** max(attempt - 1, 0)), cap)
    jitter = delay * 0.1
    return delay + random.uniform(-jitter, jitter)


class TaskProcessor:
    """Drives the processing_tasks queue through the task handlers."""

    def __init__(self, worker_id: str) -> None:
        self.worker_id = worker_id
        self._shutdown = False
        self._task: asyncio.Task | None = None
        self._qdrant_store: QdrantVectorStore | None = None
        self._bm25_store: Bm25Store | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────

    def init_deps(self) -> None:
        self._qdrant_store = QdrantVectorStore()
        self._bm25_store = Bm25Store()

    async def rebuild_bm25(self) -> None:
        """Rebuild the in-memory BM25 index from the database.

        Retries a few times on transient connection errors (common on Windows
        where cloud DB connections may flap during startup).
        """
        for attempt in range(1, 4):
            try:
                count = await self._bm25_store.rebuild_from_db()
                if count:
                    logger.info("bm25_rebuilt", count=count)
                return
            except OSError as exc:
                # Windows: [WinError 64] The specified network name is no longer available
                logger.warning(
                    "bm25_rebuild_retry",
                    attempt=attempt,
                    error=str(exc),
                )
                await asyncio.sleep(2 * attempt)
        logger.error("bm25_rebuild_failed", error="exhausted retries")

    def start(self) -> None:
        self._task = asyncio.create_task(self._run_loop(), name="task-processor")
        logger.info("task_processor_started", worker_id=self.worker_id)

    async def stop(self) -> None:
        self._shutdown = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("task_processor_stopped", worker_id=self.worker_id)

    # ── Main loop ─────────────────────────────────────────────────────

    async def _run_loop(self) -> None:
        poll_interval = settings.task_worker_poll_interval_s
        while not self._shutdown:
            try:
                processed = await self._tick()
                if processed == 0:
                    await asyncio.sleep(poll_interval)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("task_processor_tick_error")
                await asyncio.sleep(poll_interval)

    async def _tick(self) -> int:
        """Claim a batch of tasks and process them."""
        async with async_session_factory() as session:
            task_repo = ProcessingTaskRepository(session)
            try:
                claims = await task_repo.claim_pending(
                    limit=settings.task_worker_concurrency,
                    lease_minutes=settings.task_worker_lease_minutes,
                    worker_id=self.worker_id,
                )
                if not claims:
                    await session.commit()
                    return 0

                processed = 0
                for t in claims:
                    await self._process_task(session, task_repo, t)
                    await session.commit()
                    processed += 1

                return processed
            except Exception:
                await session.rollback()
                raise

    # ── Task routing ──────────────────────────────────────────────────

    async def _process_task(
        self,
        session,
        task_repo: ProcessingTaskRepository,
        task: ProcessingTaskModel,
    ) -> None:
        payload = json.loads(task.payload_json or "{}")
        handlers = {
            "document.ingest": self._handle_document_ingest,
            "document.ingest_text": self._handle_document_ingest_text,
            "document.index": self._handle_document_index,
            "vs_file.process": self._handle_vs_file_process,
        }
        handler = handlers.get(task.task_type)
        if handler is None:
            logger.error("unknown_task_type", task_type=task.task_type, task_id=task.id)
            await task_repo.mark_completed(task.id)
            return

        try:
            await handler(session, payload)
        except Exception as exc:
            next_attempt = _utcnow() + timedelta(seconds=_backoff(task.attempts))
            logger.error(
                "task_failed",
                task_id=task.id,
                task_type=task.task_type,
                error=str(exc),
                error_type=type(exc).__name__,
                traceback=traceback.format_exc(),
            )

            # Only set document to "error" when retries are exhausted
            document_id = payload.get("document_id")
            if document_id:
                repo = DocumentRepository(session)
                if task.attempts >= task.max_retries:
                    await _set_processing_status(session, repo, document_id, "error")
                else:
                    await _set_processing_status(session, repo, document_id, "retrying")

            await task_repo.mark_failed(
                task.id,
                error=f"{type(exc).__name__}: {exc}",
                next_attempt_at=next_attempt,
                max_retries=task.max_retries,
                attempts=task.attempts,
            )
            return

        # Handler succeeded — mark completed separately so a failure here doesn't
        # conflate with handler failure (which would trigger unnecessary retries).
        try:
            await task_repo.mark_completed(task.id)
        except Exception as exc:
            logger.error(
                "mark_completed_failed",
                task_id=task.id,
                task_type=task.task_type,
                error=str(exc),
                exc_info=True,
            )

    # ── Handler: document.ingest ──────────────────────────────────────

    async def _handle_document_ingest(self, session, payload: dict) -> None:
        """Download from S3, parse, chunk, store in DB, embed, index."""
        document_id = payload["document_id"]

        repo = DocumentRepository(session)
        doc = await repo.get_document(document_id)
        if doc is None:
            raise ValueError(f"Document {document_id} not found")
        if not doc.s3_key:
            raise ValueError(f"Document {document_id} missing s3_key")

        await _set_processing_status(session, repo, document_id, "processing")

        try:
            from src.storage.s3.client import S3Client

            s3 = S3Client()
            file_bytes = await s3.download(doc.s3_key)
        except Exception as exc:
            raise ValueError(f"S3 download failed: {exc}") from exc

        suffix = Path(doc.source_path or doc.title or "upload.bin").suffix or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = Path(tmp.name)

        try:
            from src.ingestion.loaders.registry import DocumentLoaderRegistry
            from src.ingestion.media.processor import MediaProcessingService

            registry = DocumentLoaderRegistry()
            loader = registry.get_loader(tmp_path)
            raw = loader.load(tmp_path)

            media_service = MediaProcessingService()
            media_result = await media_service.process(
                tmp_path, fallback_text=raw.content
            )
            raw.content = media_result.enriched_content or raw.content
            raw.metadata.update(media_result.metadata)
            raw.metadata["media_signals"] = [
                {
                    "name": s.name,
                    "source": s.source,
                    "confidence": s.confidence,
                    "content": s.content,
                    "metadata": s.metadata,
                }
                for s in media_result.signals
            ]
            raw.metadata["media_scores"] = [asdict(sc) for sc in media_result.scores]
            raw.metadata["media_summary"] = media_result.summary

            if not raw.content.strip():
                raw.content = raw.metadata.get("media_summary", "") or raw.metadata.get(
                    "title", doc.title
                )
                raw.metadata["content_fallback"] = "summary_only"

            try:
                existing_metadata = json.loads(doc.metadata_json or "{}")
            except Exception:
                existing_metadata = {}
            updated_metadata = {
                **existing_metadata,
                **raw.metadata,
                "processing_status": "processing",
            }

            await repo.update_document_metadata(
                document_id,
                metadata_json=json.dumps(updated_metadata),
                content_text=raw.content,
            )
            await repo.delete_chunks_by_document(document_id)

            from src.ingestion.chunking.parent_child import ParentChildChunker

            chunker = ParentChildChunker()
            chunks = chunker.build(raw.content, document_id)
            await repo.create_chunks(chunks)

            indexer = Indexer(session, embeddings, self._qdrant_store, self._bm25_store)
            await indexer.index_document(document_id)

            updated_metadata["processing_status"] = "processed"
            await repo.update_document_metadata(
                document_id, metadata_json=json.dumps(updated_metadata)
            )

            await _complete_vs_files_for_document(
                session, document_id, self._qdrant_store
            )

            logger.info(
                "document_ingested",
                document_id=document_id,
                chunk_count=len(chunks),
            )
        finally:
            tmp_path.unlink(missing_ok=True)

    # ── Handler: document.ingest_text ─────────────────────────────────

    async def _handle_document_ingest_text(self, session, payload: dict) -> None:
        """Chunk, store in DB, embed, and index a text document."""
        document_id = payload["document_id"]
        text = payload["content_text"]

        repo = DocumentRepository(session)
        await _set_processing_status(session, repo, document_id, "processing")

        from src.ingestion.chunking.parent_child import ParentChildChunker

        chunker = ParentChildChunker()
        # Delete existing chunks first so retry is idempotent
        await repo.delete_chunks_by_document(document_id)
        chunks = chunker.build(text, document_id)
        await repo.create_chunks(chunks)

        indexer = Indexer(session, embeddings, self._qdrant_store, self._bm25_store)
        await indexer.index_document(document_id)

        # Update metadata
        doc = await repo.get_document(document_id)
        if doc:
            try:
                meta = json.loads(doc.metadata_json or "{}")
            except Exception:
                meta = {}
            meta["processing_status"] = "processed"
            await repo.update_document_metadata(
                document_id,
                metadata_json=json.dumps(meta),
                content_text=text,
            )

        await _complete_vs_files_for_document(session, document_id, self._qdrant_store)

        logger.info(
            "text_ingested",
            document_id=document_id,
            chunk_count=len(chunks),
        )

    # ── Handler: document.index ───────────────────────────────────────

    async def _handle_document_index(self, session, payload: dict) -> None:
        """Re-index an already-chunked document."""
        document_id = payload["document_id"]

        indexer = Indexer(session, embeddings, self._qdrant_store, self._bm25_store)
        await indexer.index_document(document_id)

        logger.info("document_reindexed", document_id=document_id)

    # ── Handler: vs_file.process ──────────────────────────────────────

    async def _handle_vs_file_process(self, session, payload: dict) -> None:
        """Tag chunks with vector_store_id. Mirrors old VectorStoreWorker._process_file."""
        from src.database.repositories import VectorStoreFileRepository
        from src.vector_stores.repository import VectorStoreRepository

        vf_id = payload["vector_store_file_id"]
        vf_repo = VectorStoreFileRepository(session)
        vs_repo = VectorStoreRepository(session)
        doc_repo = DocumentRepository(session)

        vf = await vf_repo.get(vf_id)
        if vf is None:
            return

        store = await vs_repo.get(vf.vector_store_id)
        doc = await doc_repo.get_document(vf.source_document_id)

        if store is None or doc is None:
            await vf_repo.mark_failed(
                vf_id, failure_reason="missing_vector_store_or_document"
            )
            await vf_repo.update_store_file_counts(vf.vector_store_id)
            return

        # Check if source document is ready
        doc_status = "uploaded"
        try:
            meta = json.loads(getattr(doc, "metadata_json", None) or "{}")
            doc_status = meta.get("processing_status", "uploaded")
        except Exception:
            pass

        if doc_status not in ("processed", "completed", "error"):
            existing_chunks = await doc_repo.get_chunks_by_document(
                vf.source_document_id
            )
            if not existing_chunks:
                # Not ready yet — release lock for retry
                await session.execute(
                    sa_update(VectorStoreFileModel)
                    .where(
                        VectorStoreFileModel.id == vf.id,
                        VectorStoreFileModel.status == "processing",
                    )
                    .values(status="pending", locked_at=None, locked_by=None)
                )
                return

        if doc_status == "error":
            await vf_repo.mark_failed(
                vf_id, failure_reason="source_document_processing_failed"
            )
            await vf_repo.update_store_file_counts(vf.vector_store_id)
            return

        # Tag chunks
        await vf_repo.update_status(vf_id, status="chunking")
        await doc_repo.update_chunks_vector_store_id(
            vf.source_document_id, vf.vector_store_id
        )

        if self._qdrant_store is not None:
            await self._qdrant_store.update_chunks_vector_store_id(
                vf.source_document_id, vf.vector_store_id
            )

        await vf_repo.mark_completed(vf_id)
        await vf_repo.update_store_file_counts(vf.vector_store_id)
        # Update batch counts + status if this file belongs to a batch
        if vf.batch_id is not None:
            from src.vector_stores.repository import (
                VectorStoreFileBatchRepository,
            )
            vfb_repo = VectorStoreFileBatchRepository(session)
            await vfb_repo.update_counts_and_status(vf.batch_id)

        logger.info(
            "vs_file_processed",
            file_id=vf_id,
            store_id=vf.vector_store_id,
            document_id=vf.source_document_id,
        )


# ── Helpers ──────────────────────────────────────────────────────────


async def _set_processing_status(
    session, repo: DocumentRepository, document_id: str, status: str
) -> None:
    row = await repo.get_document(document_id)
    if row is None:
        return
    try:
        meta = json.loads(row.metadata_json or "{}")
    except Exception:
        meta = {}
    meta["processing_status"] = status
    await repo.update_document_metadata(document_id, metadata_json=json.dumps(meta))


async def _complete_vs_files_for_document(
    session, document_id: str, qdrant_store: QdrantVectorStore | None
) -> None:
    """Complete pending VS files. Raises on failure so the task retries."""
    from src.database.repositories import VectorStoreFileRepository
    from src.vector_stores.repository import VectorStoreFileBatchRepository

    vf_repo = VectorStoreFileRepository(session)
    doc_repo = DocumentRepository(session)

    affected_store_ids = await vf_repo.complete_pending_for_document(document_id)
    if not affected_store_ids:
        return

    for store_id in affected_store_ids:
        await doc_repo.update_chunks_vector_store_id(document_id, store_id)
        if qdrant_store is not None:
            await qdrant_store.update_chunks_vector_store_id(document_id, store_id)
        await vf_repo.update_store_file_counts(store_id)
        # Also update batch counts for any batches containing this document's files
        from sqlalchemy import select as sa_select
        from src.database.models import VectorStoreFileModel

        batch_result = await session.execute(
            sa_select(VectorStoreFileModel.batch_id).where(
                VectorStoreFileModel.source_document_id == document_id,
                VectorStoreFileModel.vector_store_id == store_id,
                VectorStoreFileModel.batch_id.is_not(None),
            ).distinct()
        )
        batch_ids = [row[0] for row in batch_result.fetchall()]
        vfb_repo = VectorStoreFileBatchRepository(session)
        for batch_id in batch_ids:
            await vfb_repo.update_counts_and_status(batch_id)

    logger.info(
        "vs_files_completed_for_document",
        document_id=document_id,
        store_count=len(affected_store_ids),
    )
