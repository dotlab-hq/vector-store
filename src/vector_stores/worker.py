"""Background worker that processes pending vector_store_files.

Runs as an asyncio task in the FastAPI lifespan (in-process) or as a standalone
entrypoint (``python -m apps.worker``). Uses PostgreSQL ``SELECT ... FOR UPDATE
SKIP LOCKED`` to safely claim files across multiple workers.
"""
from __future__ import annotations

import asyncio
import random
from datetime import datetime, timedelta

from src.config import settings
from src.database.session import async_session_factory
from src.observability.logging import get_logger
from src.vector_stores.repository import (
    VectorStoreFileRepository,
    VectorStoreRepository,
)

logger = get_logger()


def _utcnow() -> datetime:
    """Naive UTC datetime matching SQLAlchemy's func.now()."""
    return datetime.utcnow()


def _backoff(attempt: int) -> float:
    """Exponential backoff with jitter: min(base * 2^(attempt-1), cap) +/- 10%."""
    base = settings.vector_store_retry_base_s
    cap = settings.vector_store_retry_cap_s
    delay = min(base * (2 ** max(attempt - 1, 0)), cap)
    jitter = delay * 0.1
    return delay + random.uniform(-jitter, jitter)


class VectorStoreWorker:
    """Drives vector_store_files through the processing pipeline."""

    def __init__(self) -> None:
        self._shutdown = False
        self._task: asyncio.Task | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        self._task = asyncio.create_task(self._run_loop(), name="vs-worker")
        logger.info("vs_worker_started")

    async def stop(self) -> None:
        self._shutdown = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("vs_worker_stopped")

    # ── Main loop ─────────────────────────────────────────────────────

    async def _run_loop(self) -> None:
        poll_interval = settings.vector_store_worker_poll_interval_s
        while not self._shutdown:
            try:
                processed = await self._tick()
                if processed == 0:
                    await asyncio.sleep(poll_interval)
            except asyncio.CancelledError:
                raise
            except Exception:
                await asyncio.sleep(poll_interval)

    async def _tick(self) -> int:
        """Claim a batch and process them. Returns count processed."""
        from src.database.repositories import DocumentRepository

        lease_minutes = settings.vector_store_worker_lease_minutes
        async with async_session_factory() as session:
            vs_repo = VectorStoreRepository(session)
            vf_repo = VectorStoreFileRepository(session)
            doc_repo = DocumentRepository(session)

            try:
                claims = await vf_repo.claim_pending(
                    limit=settings.vector_store_worker_concurrency,
                    lease_minutes=lease_minutes,
                )
                if not claims:
                    await session.commit()
                    return 0

                # Load stores in one batch
                store_ids = {f.vector_store_id for f in claims}
                stores: dict[str, object] = {}
                for sid in store_ids:
                    store = await vs_repo.get(sid)
                    if store is not None:
                        stores[sid] = store

                doc_ids = [f.source_document_id for f in claims]
                documents = await doc_repo.get_documents_by_ids(doc_ids)

                processed = 0
                for vf in claims:
                    store = stores.get(vf.vector_store_id)
                    doc = documents.get(vf.source_document_id)
                    if store is None or doc is None:
                        logger.warning(
                            "vs_worker_skip_missing_refs",
                            file_id=vf.id,
                            store_id=vf.vector_store_id,
                            doc_id=vf.source_document_id,
                            found_store=store is not None,
                            found_doc=doc is not None,
                        )
                        await vf_repo.mark_failed(
                            vf.id,
                            failure_reason="missing_vector_store_or_document",
                            next_attempt_at=_utcnow() + timedelta(seconds=_backoff(vf.attempts)),
                        )
                        await session.commit()
                        processed += 1
                        continue

                    await self._process_file(session, vf_repo, doc_repo, vf, doc, store)
                    await session.commit()
                    processed += 1

                await session.commit()
                return processed
            except Exception:
                await session.rollback()
                raise

    # ── Per-file processing ───────────────────────────────────────────

    async def _process_file(
        self,
        session: object,
        vf_repo: VectorStoreFileRepository,
        doc_repo: object,
        vf: object,
        doc: object,
        store: object,
    ) -> None:
        """Tag DB + Qdrant chunks with the vector store ID, then complete.

        This worker no longer re-chunks / re-embeds / re-indexes — that is
        the upload pipeline's job.  If the source document hasn't finished
        processing yet, we release the lock and defer rather than spin.
        """
        now = _utcnow()

        try:
            # ── Gate: wait for the upload pipeline to finish ──────
            doc_status = "uploaded"
            try:
                import json
                meta = json.loads(getattr(doc, "metadata_json", None) or "{}")
                doc_status = meta.get("processing_status", "uploaded")
            except Exception:
                pass

            if doc_status not in ("processed", "completed", "error"):
                # Document still being processed — check if chunks exist
                # anyway (the upload pipeline may have created them but not
                # yet set the metadata flag).
                existing_chunks = await doc_repo.get_chunks_by_document(
                    vf.source_document_id
                )
                if not existing_chunks:
                    # Truly not ready — release the lock so the cron
                    # can re-claim later.  Don't burn a retry attempt.
                    from sqlalchemy import update as sa_update
                    from src.database.models import VectorStoreFileModel as VF
                    await session.execute(
                        sa_update(VF)
                        .where(VF.id == vf.id, VF.status == "processing")
                        .values(status="pending", locked_at=None, locked_by=None)
                    )
                    await session.commit()
                    return
                # Chunks exist — proceed to tag them (the upload
                # pipeline created them, we just need to wire up the
                # vector_store_id).

            if doc_status == "error":
                await vf_repo.mark_failed(
                    vf.id, failure_reason="source_document_processing_failed"
                )
                await vf_repo.update_store_file_counts(vf.vector_store_id)
                await session.commit()
                return

            # ── Tag chunks with vector_store_id ──────────────────
            await vf_repo.update_status(vf.id, status="chunking")

            from apps.api.dependencies import qdrant_store

            await doc_repo.update_chunks_vector_store_id(
                vf.source_document_id, vf.vector_store_id
            )
            if qdrant_store is not None:
                await qdrant_store.update_chunks_vector_store_id(
                    vf.source_document_id, vf.vector_store_id
                )

            # ── Complete ─────────────────────────────────────────
            await vf_repo.mark_completed(vf.id)
            await vf_repo.update_store_file_counts(vf.vector_store_id)

            logger.info(
                "vs_file_processing_completed",
                file_id=vf.id,
                store_id=vf.vector_store_id,
                document_id=vf.source_document_id,
            )

        except Exception as exc:
            import traceback

            error_msg = f"{type(exc).__name__}: {exc}"
            logger.error(
                "vs_file_processing_failed",
                file_id=vf.id,
                store_id=vf.vector_store_id,
                document_id=vf.source_document_id,
                error=error_msg,
                traceback=traceback.format_exc(),
            )
            next_attempt = now + timedelta(seconds=_backoff(vf.attempts))
            try:
                await vf_repo.mark_failed(
                    vf.id,
                    failure_reason=error_msg,
                    next_attempt_at=next_attempt,
                )
                await vf_repo.update_store_file_counts(vf.vector_store_id)
            except Exception:
                await session.rollback()
                await vf_repo.mark_failed(
                    vf.id,
                    failure_reason=error_msg,
                    next_attempt_at=next_attempt,
                )
                await vf_repo.update_store_file_counts(vf.vector_store_id)
