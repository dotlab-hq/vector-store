"""arq worker configuration — registers task handler functions.

Run with: ``arq apps.worker.arq_settings.WorkerSettings``

Each handler mirrors the corresponding ``TaskProcessor._handle_*`` method
but is a plain async function with ``ctx`` as the first parameter.
Database sessions are created per-job inside each handler.
"""

from __future__ import annotations

import functools
import json
import tempfile
import traceback
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Any

from arq.connections import RedisSettings

from src.config import settings
from src.database.models import VectorStoreFileModel
from src.database.repositories import DocumentRepository
from src.database.session import async_session_factory
from src.indexing.bm25.bm25_store import Bm25Store
from src.indexing.embeddings import embeddings
from src.indexing.indexer import Indexer
from src.indexing.qdrant.qdrant_store import QdrantVectorStore
from src.observability.logging import get_logger
from src.shared.events import get_redis_settings

logger = get_logger()

# ── Shared dependencies (lazily initialised on first job) ─────────────

_qdrant_store: QdrantVectorStore | None = None
_bm25_store: Bm25Store | None = None


def _ensure_deps() -> tuple[QdrantVectorStore, Bm25Store]:
    global _qdrant_store, _bm25_store
    if _qdrant_store is None:
        _qdrant_store = QdrantVectorStore()
    if _bm25_store is None:
        _bm25_store = Bm25Store()
    return _qdrant_store, _bm25_store


# ── Helpers ──────────────────────────────────────────────────────────


def _backoff(attempt: int) -> float:
    """Exponential backoff with jitter."""
    import random

    base = settings.task_worker_retry_base_s
    cap = settings.task_worker_retry_cap_s
    delay = min(base * (2 ** max(attempt - 1, 0)), cap)
    jitter = delay * 0.1
    return delay + random.uniform(-jitter, jitter)


async def _set_processing_status(
    document_id: str, status: str
) -> None:
    """Set the processing_status in a document's metadata."""
    import json as _json

    async with async_session_factory() as session:
        repo = DocumentRepository(session)
        row = await repo.get_document(document_id)
        if row is None:
            return
        try:
            meta = _json.loads(row.metadata_json or "{}")
        except Exception:
            meta = {}
        meta["processing_status"] = status
        await repo.update_document_metadata(
            document_id, metadata_json=_json.dumps(meta)
        )
        await session.commit()


async def _complete_vs_files_for_document(
    document_id: str, qdrant_store: QdrantVectorStore | None
) -> None:
    """Complete pending VS files after a document is ingested."""
    async with async_session_factory() as session:
        from src.database.repositories import VectorStoreFileRepository

        vf_repo = VectorStoreFileRepository(session)
        doc_repo = DocumentRepository(session)

        affected_store_ids = await vf_repo.complete_pending_for_document(document_id)
        if not affected_store_ids:
            await session.commit()
            return

        for store_id in affected_store_ids:
            await doc_repo.update_chunks_vector_store_id(document_id, store_id)
            if qdrant_store is not None:
                await qdrant_store.update_chunks_vector_store_id(
                    document_id, store_id
                )
            await vf_repo.update_store_file_counts(store_id)

        await session.commit()
        logger.info(
            "vs_files_completed_for_document",
            document_id=document_id,
            store_count=len(affected_store_ids),
        )


# ── Crash-safe decorator ─────────────────────────────────────────────


def _crash_safe(
    handler: Any,
) -> Any:
    """Wrap a worker handler so that exceptions set ``processing_status="error"``.

    Catches any exception from the handler, attempts to mark the document/VF
    as failed, logs the full traceback, and re-raises so arq can schedule a
    retry (up to ``max_tries``).
    """

    @functools.wraps(handler)
    async def wrapper(ctx: dict[str, Any], **kwargs: Any) -> None:
        try:
            return await handler(ctx, **kwargs)
        except Exception:
            logger.exception("handler_crashed", handler=handler.__name__)
            document_id: str | None = kwargs.get("document_id")
            if document_id:
                try:
                    await _set_processing_status(document_id, "error")
                except Exception:
                    logger.exception(
                        "failed_to_set_error_status", document_id=document_id
                    )
            vf_id: str | None = kwargs.get("vector_store_file_id")
            if vf_id:
                try:
                    async with async_session_factory() as session:
                        from src.database.repositories import (
                            VectorStoreFileRepository,
                        )

                        vf_repo = VectorStoreFileRepository(session)
                        await vf_repo.mark_failed(
                            vf_id, failure_reason="handler_crashed"
                        )
                        await session.commit()
                except Exception:
                    logger.exception(
                        "failed_to_mark_vf_failed", file_id=vf_id
                    )
            raise

    return wrapper


# ── Worker functions ─────────────────────────────────────────────────


@_crash_safe
async def document_ingest(
    ctx: dict[str, Any],
    *,
    document_id: str,
    s3_key: str | None = None,
    purpose: str = "assistants",
) -> None:
    """Download from S3, parse, chunk, store in DB, embed, index.

    Mirrors ``TaskProcessor._handle_document_ingest``.
    """
    qdrant_store, bm25_store = _ensure_deps()

    async with async_session_factory() as session:
        repo = DocumentRepository(session)
        doc = await repo.get_document(document_id)
        if doc is None:
            raise ValueError(f"Document {document_id} not found")
        if not doc.s3_key:
            raise ValueError(f"Document {document_id} missing s3_key")

        meta = json.loads(doc.metadata_json or "{}")
        meta["processing_status"] = "processing"
        await repo.update_document_metadata(
            document_id, metadata_json=json.dumps(meta)
        )

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

            # ── PDF Diagram Extraction ──────────────────────────────
            diagram_chunks: list[Chunk] = []
            suffix_lower = suffix.lower()
            if suffix_lower == ".pdf" and settings.pdf_extract_diagrams:
                try:
                    from src.ingestion.parsers.pdf_diagram import PDFDiagramExtractor

                    s3_client = S3Client()
                    extractor = PDFDiagramExtractor()
                    pages = await extractor.extract_all_pages(tmp_path)

                    for page_data in pages:
                        page_number = page_data["page_number"]
                        for diagram in page_data.get("diagrams", []):
                            image_bytes = diagram.get("image_bytes")
                            description = diagram.get("description", "")
                            region_idx = diagram.get("region_index", 0)
                            mime_type = diagram.get("mime_type", "image/png")

                            if not image_bytes or not description:
                                continue

                            s3_key = (
                                f"diagrams/{document_id}/"
                                f"page_{page_number}_region_{region_idx}.png"
                            )

                            await s3_client.upload(
                                key=s3_key,
                                data=image_bytes,
                                content_type=mime_type,
                            )
                            logger.info(
                                "diagram_uploaded",
                                document_id=document_id,
                                s3_key=s3_key,
                                page=page_number,
                                region=region_idx,
                            )

                            diagram_chunks.append(
                                Chunk(
                                    id=f"{document_id}_diag_p{page_number}_r{region_idx}",
                                    document_id=document_id,
                                    content=description,
                                    page_number=page_number,
                                    position=100_000 + page_number * 100 + region_idx,
                                    section="diagram",
                                    image_url=s3_key,
                                )
                            )

                    if diagram_chunks:
                        await repo.create_chunks(diagram_chunks)
                        chunks.extend(diagram_chunks)
                        logger.info(
                            "diagram_chunks_created",
                            document_id=document_id,
                            count=len(diagram_chunks),
                        )
                except ImportError:
                    logger.debug("pymupdf_not_available_skipping_diagrams")
                except Exception as exc:
                    logger.warning(
                        "diagram_extraction_failed",
                        document_id=document_id,
                        error=str(exc),
                        exc_info=True,
                    )

            indexer = Indexer(session, embeddings, qdrant_store, bm25_store)
            await indexer.index_document(document_id)

            updated_metadata["processing_status"] = "processed"
            await repo.update_document_metadata(
                document_id, metadata_json=json.dumps(updated_metadata)
            )
            await session.commit()

            await _complete_vs_files_for_document(document_id, qdrant_store)

            logger.info(
                "document_ingested",
                document_id=document_id,
                chunk_count=len(chunks),
            )
        finally:
            tmp_path.unlink(missing_ok=True)


@_crash_safe
async def document_ingest_text(
    ctx: dict[str, Any],
    *,
    document_id: str,
    content_text: str,
    title: str = "Untitled",
) -> None:
    """Chunk, store in DB, embed, and index a text document.

    Mirrors ``TaskProcessor._handle_document_ingest_text``.
    """
    qdrant_store, bm25_store = _ensure_deps()

    async with async_session_factory() as session:
        repo = DocumentRepository(session)
        meta_row = await repo.get_document(document_id)

        if meta_row is None:
            raise ValueError(f"Document {document_id} not found")

        # Set processing
        try:
            meta = json.loads(meta_row.metadata_json or "{}")
        except Exception:
            meta = {}
        meta["processing_status"] = "processing"
        await repo.update_document_metadata(
            document_id, metadata_json=json.dumps(meta)
        )

        from src.ingestion.chunking.parent_child import ParentChildChunker

        chunker = ParentChildChunker()
        await repo.delete_chunks_by_document(document_id)
        chunks = chunker.build(content_text, document_id)
        await repo.create_chunks(chunks)

        indexer = Indexer(session, embeddings, qdrant_store, bm25_store)
        await indexer.index_document(document_id)

        doc = await repo.get_document(document_id)
        if doc:
            try:
                m = json.loads(doc.metadata_json or "{}")
            except Exception:
                m = {}
            m["processing_status"] = "processed"
            await repo.update_document_metadata(
                document_id,
                metadata_json=json.dumps(m),
                content_text=content_text,
            )

        await session.commit()

        await _complete_vs_files_for_document(document_id, qdrant_store)

        logger.info(
            "text_ingested",
            document_id=document_id,
            chunk_count=len(chunks),
        )


@_crash_safe
async def document_index(
    ctx: dict[str, Any],
    *,
    document_id: str,
) -> None:
    """Re-index an already-chunked document.

    Mirrors ``TaskProcessor._handle_document_index``.
    """
    qdrant_store, bm25_store = _ensure_deps()

    async with async_session_factory() as session:
        indexer = Indexer(session, embeddings, qdrant_store, bm25_store)
        await indexer.index_document(document_id)
        await session.commit()

    logger.info("document_reindexed", document_id=document_id)


@_crash_safe
async def vs_file_process(
    ctx: dict[str, Any],
    *,
    vector_store_file_id: str,
) -> None:
    """Tag chunks with vector_store_id.

    Mirrors ``TaskProcessor._handle_vs_file_process``.
    """
    qdrant_store, _bm25 = _ensure_deps()

    async with async_session_factory() as session:
        from sqlalchemy import update as sa_update

        from src.database.repositories import VectorStoreFileRepository
        from src.database.repositories import VectorStoreRepository

        vf_repo = VectorStoreFileRepository(session)
        vs_repo = VectorStoreRepository(session)
        doc_repo = DocumentRepository(session)

        vf = await vf_repo.get(vector_store_file_id)
        if vf is None:
            return

        store = await vs_repo.get(vf.vector_store_id)
        doc = await doc_repo.get_document(vf.source_document_id)

        if store is None or doc is None:
            await vf_repo.mark_failed(
                vf.id, failure_reason="missing_vector_store_or_document"
            )
            await vf_repo.update_store_file_counts(vf.vector_store_id)
            await session.commit()
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
                # Not ready — release lock for retry
                await session.execute(
                    sa_update(VectorStoreFileModel)
                    .where(
                        VectorStoreFileModel.id == vf.id,
                        VectorStoreFileModel.status == "processing",
                    )
                    .values(status="pending", locked_at=None, locked_by=None)
                )
                await session.commit()
                return

        if doc_status == "error":
            await vf_repo.mark_failed(
                vf.id, failure_reason="source_document_processing_failed"
            )
            await vf_repo.update_store_file_counts(vf.vector_store_id)
            await session.commit()
            return

        # Tag chunks
        await vf_repo.update_status(vf.id, status="chunking")
        await doc_repo.update_chunks_vector_store_id(
            vf.source_document_id, vf.vector_store_id
        )

        if qdrant_store is not None:
            await qdrant_store.update_chunks_vector_store_id(
                vf.source_document_id, vf.vector_store_id
            )

        await vf_repo.mark_completed(vf.id)
        await vf_repo.update_store_file_counts(vf.vector_store_id)
        await session.commit()

        logger.info(
            "vs_file_processed",
            file_id=vf.id,
            store_id=vf.vector_store_id,
            document_id=vf.source_document_id,
        )


# ── arq WorkerSettings ───────────────────────────────────────────────


class WorkerSettings:
    """arq worker configuration.

    Run with: ``arq apps.worker.arq_settings.WorkerSettings``
    """

    functions = [document_ingest, document_ingest_text, document_index, vs_file_process]
    redis_settings: RedisSettings = get_redis_settings()
    queue_name = "default"

    # Worker tuning
    max_jobs = settings.task_worker_concurrency
    poll_delay = settings.task_worker_poll_interval_s

    # Retry settings — arq handles retries natively
    max_tries = settings.task_worker_retry_max
    retry_delay = settings.task_worker_retry_base_s

    # Health check interval (seconds) — how often arq checks for new jobs
    health_check_interval = 30
