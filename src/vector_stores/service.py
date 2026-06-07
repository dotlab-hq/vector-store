"""Vector store service — orchestration for create, attach, search, cancel."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from uuid import uuid4

from sqlalchemy import update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession


def _utcnow() -> datetime:
    """Naive UTC — matches SQLAlchemy's func.now()."""
    return datetime.utcnow()

from apps.api.schemas.vector_stores import (
    AutoChunkingStrategy,
    ChunkingStrategy,
    ContentBlock,
    CreateVectorStoreFileBatchRequest,
    CreateVectorStoreFileRequest,
    CreateVectorStoreRequest,
    DeleteResponse,
    ExpiresAfter,
    FileContentItem,
    FileContentResponse,
    FileCounts,
    LastError,
    ListVectorStoreFilesResponse,
    ListVectorStoresResponse,
    PerFileConfig,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    StaticChunkingStrategy,
    StaticChunkingStrategyConfig,
    UpdateVectorStoreFileRequest,
    UpdateVectorStoreRequest,
    VectorStoreFileBatchFileCounts,
    VectorStoreFileBatchObject,
    VectorStoreFileObject,
    VectorStoreObject,
    chunking_strategy_to_internal,
)
from src.database.models import (
    VectorStoreFileBatchModel,
    VectorStoreFileModel,
    VectorStoreModel,
)
from src.database.repositories import DocumentRepository
from src.observability.logging import get_logger
from src.vector_stores.filter_eval import compile_predicate
from src.vector_stores.models import (
    VECTOR_STORE_FILE_BATCH_ID_PREFIX,
    VECTOR_STORE_FILE_ID_PREFIX,
    VECTOR_STORE_ID_PREFIX,
    TERMINAL_FILE_STATUSES,
)
from src.vector_stores.repository import (
    VectorStoreFileBatchRepository,
    VectorStoreFileRepository,
    VectorStoreRepository,
)

logger = get_logger()


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------


def _new_vector_store_id() -> str:
    return f"{VECTOR_STORE_ID_PREFIX}{uuid4().hex[:24]}"


def _new_file_id() -> str:
    return f"{VECTOR_STORE_FILE_ID_PREFIX}{uuid4().hex[:24]}"


def _new_batch_id() -> str:
    return f"{VECTOR_STORE_FILE_BATCH_ID_PREFIX}{uuid4().hex[:24]}"


def _derive_chunking_strategy(
    strategy_name: str, chunk_size: int | None, chunk_overlap: int | None
) -> ChunkingStrategy:
    """Reconstruct the OpenAI ChunkingStrategy from the store fields."""
    if strategy_name == "static" and chunk_size is not None:
        return StaticChunkingStrategy(
            static=StaticChunkingStrategyConfig(
                max_chunk_size_tokens=chunk_size,
                chunk_overlap_tokens=chunk_overlap or 400,
            )
        )
    return AutoChunkingStrategy()


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def _to_vector_store_object(
    store: VectorStoreModel,
    counts: dict[str, int] | None = None,
) -> VectorStoreObject:
    fc = counts or json.loads(store.file_counts_json or "{}")
    file_counts = FileCounts(
        in_progress=fc.get("in_progress", 0),
        completed=fc.get("completed", 0),
        cancelled=fc.get("cancelled", 0),
        failed=fc.get("failed", 0),
        total=fc.get("total", 0),
    )
    md = json.loads(store.metadata_json or "{}")
    metadata = {str(k): str(v) for k, v in md.items()}

    expires_after = None
    expires_at_unix = None
    if store.expires_after_days is not None:
        expires_after = ExpiresAfter(anchor="last_active_at", days=store.expires_after_days)
    if store.expires_at is not None:
        expires_at_unix = int(store.expires_at.timestamp())

    return VectorStoreObject(
        id=store.id,
        name=store.name or "",
        status=store.status or "in_progress",
        file_counts=file_counts,
        last_active_at=int(store.last_active_at.timestamp()) if store.last_active_at else 0,
        created_at=int(store.created_at.timestamp()) if store.created_at else 0,
        bytes=store.usage_bytes or 0,
        metadata=metadata,
        expires_after=expires_after,
        expires_at=expires_at_unix,
    )


def _map_internal_status_to_oai(status: str) -> str:
    """Map our internal statuses (pending, processing, chunking, etc.) to OpenAI's
    coarser 4-state set: in_progress, completed, failed, cancelled.
    """
    if status in ("completed",):
        return "completed"
    if status in ("cancelled",):
        return "cancelled"
    if status in ("failed",):
        return "failed"
    # pending, processing, chunking, embedding, indexing all map to in_progress
    return "in_progress"


def _classify_failure_reason(reason: str | None) -> str:
    """Classify a failure reason string into an OpenAI error code."""
    if reason is None:
        return "server_error"
    reason_lower = reason.lower()
    if "unsupported" in reason_lower or "no loader" in reason_lower:
        return "unsupported_file"
    if "invalid" in reason_lower or "corrupt" in reason_lower:
        return "invalid_file"
    return "server_error"


def _to_vector_store_file_object(
    vf: VectorStoreFileModel,
    store_id: str | None = None,
    chunking_strategy: ChunkingStrategy | None = None,
) -> VectorStoreFileObject:
    if chunking_strategy is None:
        # Default to auto when not specified
        chunking_strategy = AutoChunkingStrategy()
    attributes: dict = {}
    try:
        attributes = json.loads(vf.attributes_json or "{}")
    except (json.JSONDecodeError, TypeError):
        attributes = {}

    last_error: LastError | None = None
    if vf.failure_reason:
        last_error = LastError(
            code=_classify_failure_reason(vf.failure_reason),  # type: ignore[arg-type]
            message=vf.failure_reason,
        )

    oai_status = _map_internal_status_to_oai(vf.status or "pending")

    return VectorStoreFileObject(
        id=vf.id,
        vector_store_id=store_id or vf.vector_store_id,
        status=oai_status,  # type: ignore[arg-type]
        last_error=last_error,
        created_at=int(vf.created_at.timestamp()) if vf.created_at else 0,
        bytes=vf.bytes or 0,
        usage_bytes=vf.bytes or 0,
        attributes=attributes,
        chunking_strategy=chunking_strategy,
    )


def _to_vector_store_file_batch_object(
    batch: VectorStoreFileBatchModel,
    counts: dict[str, int] | None = None,
) -> VectorStoreFileBatchObject:
    fc = counts or json.loads(batch.file_counts_json or "{}")
    file_counts = VectorStoreFileBatchFileCounts(
        in_progress=fc.get("in_progress", 0),
        completed=fc.get("completed", 0),
        cancelled=fc.get("cancelled", 0),
        failed=fc.get("failed", 0),
        total=fc.get("total", 0),
    )
    return VectorStoreFileBatchObject(
        id=batch.id,
        vector_store_id=batch.vector_store_id,
        status=batch.status or "in_progress",  # type: ignore[arg-type]
        file_counts=file_counts,
        created_at=int(batch.created_at.timestamp()) if batch.created_at else 0,
    )


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class VectorStoreService:
    """High-level orchestration. Takes an ``AsyncSession`` per call."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.vs_repo = VectorStoreRepository(session)
        self.vf_repo = VectorStoreFileRepository(session)
        self.vfb_repo = VectorStoreFileBatchRepository(session)
        self.doc_repo = DocumentRepository(session)

    # ── Create vector store ──────────────────────────────────────────

    async def create_store(
        self, request: CreateVectorStoreRequest
    ) -> VectorStoreObject:
        strategy_name, chunk_size, chunk_overlap = chunking_strategy_to_internal(
            request.chunking_strategy
        )

        expires_at = None
        expires_after_days = None
        if request.expires_after is not None:
            expires_after_days = request.expires_after.days
            expires_at = _utcnow() + timedelta(
                days=expires_after_days
            )

        metadata_json = json.dumps(request.metadata or {})
        store = VectorStoreModel(
            id=_new_vector_store_id(),
            name=request.name or "",
            status="in_progress",
            metadata_json=metadata_json,
            chunking_strategy=strategy_name,
            chunk_size_tokens=chunk_size,
            chunk_overlap_tokens=chunk_overlap,
            expires_at=expires_at,
            expires_after_days=expires_after_days,
        )
        await self.vs_repo.create(store)
        await self.session.flush()

        # Attach initial files (if any)
        if request.file_ids:
            await self._attach_files(
                store.id, request.file_ids, request.chunking_strategy
            )

        await self.session.commit()
        # Re-fetch to get updated file_counts
        store = await self.vs_repo.get(store.id)  # type: ignore[assignment]
        return _to_vector_store_object(store)

    async def _is_document_ready(self, doc: "DocumentModel") -> bool:
        """Check if a document is ready to search — chunks exist AND are indexed."""
        # Fast path: metadata flag
        try:
            meta = json.loads(doc.metadata_json or "{}")
            if meta.get("processing_status") in ("processed", "completed"):
                return True
        except (json.JSONDecodeError, TypeError):
            pass
        # Authoritative path: do chunks actually exist?
        chunks = await self.doc_repo.get_chunks_by_document(doc.id)
        return len(chunks) > 0

    async def _attach_files(
        self,
        store_id: str,
        file_ids: list[str],
        chunking_strategy: ChunkingStrategy | None,
    ) -> list[VectorStoreFileObject]:
        results: list[VectorStoreFileObject] = []
        # Get the store once to derive the chunking strategy
        store = await self.vs_repo.get(store_id)
        chunking = _derive_chunking_strategy(
            store.chunking_strategy if store else "auto",
            store.chunk_size_tokens if store else None,
            store.chunk_overlap_tokens if store else None,
        )
        for fid in file_ids:
            doc = await self.doc_repo.get_document(fid)
            if doc is None:
                logger.warning(
                    "vs_attach_missing_document", file_id=fid, store_id=store_id
                )
                continue
            # Idempotent: check if already attached
            existing = await self.vf_repo.get_by_store_and_document(store_id, fid)
            if existing is not None:
                results.append(_to_vector_store_file_object(existing, store_id, chunking))
                continue
            # If the source document is already fully processed (ingested/chunked),
            # mark the vector-store file as completed immediately — there is nothing
            # left for a background worker to do.  We also need to tag the Qdrant
            # chunks with the store ID so search_in_stores can find them.
            initial_status = "completed" if await self._is_document_ready(doc) else "pending"
            vf = VectorStoreFileModel(
                id=_new_file_id(),
                vector_store_id=store_id,
                source_document_id=fid,
                status=initial_status,
                bytes=doc.bytes or 0,
                completed_at=_utcnow() if initial_status == "completed" else None,
            )
            await self.vf_repo.create(vf)
            if initial_status == "completed":
                # Tag DB rows
                await self.doc_repo.update_chunks_vector_store_id(fid, store_id)
                # Tag Qdrant payloads
                from apps.api.dependencies import qdrant_store
                if qdrant_store is not None:
                    await qdrant_store.update_chunks_vector_store_id(fid, store_id)
            results.append(_to_vector_store_file_object(vf, store_id, chunking))
        # Recompute counts
        await self.vf_repo.update_store_file_counts(store_id)
        return results

    # ── Get / list / update / delete ─────────────────────────────────

    async def get_store(self, store_id: str) -> VectorStoreObject | None:
        store = await self.vs_repo.get(store_id)
        if store is None:
            return None
        counts = json.loads(store.file_counts_json or "{}")
        return _to_vector_store_object(store, counts)

    async def list_stores(
        self, *, limit: int = 20, after_id: str | None = None
    ) -> ListVectorStoresResponse:
        stores = await self.vs_repo.list_all(limit=limit, after_id=after_id)
        has_more = len(stores) > limit
        stores = stores[:limit]
        data = [_to_vector_store_object(s) for s in stores]
        return ListVectorStoresResponse(
            data=data,
            has_more=has_more,
            first_id=data[0].id if data else None,
            last_id=data[-1].id if data else None,
        )

    async def update_store(
        self, store_id: str, request: UpdateVectorStoreRequest
    ) -> VectorStoreObject | None:
        store = await self.vs_repo.get(store_id)
        if store is None:
            return None

        updates: dict = {}
        if request.name is not None:
            updates["name"] = request.name
        if request.metadata is not None:
            updates["metadata_json"] = json.dumps(request.metadata)
        if request.expires_after is not None:
            updates["expires_after_days"] = request.expires_after.days
            updates["expires_at"] = _utcnow() + timedelta(
                days=request.expires_after.days
            )
        if updates:
            await self.vs_repo.update(store_id, **updates)
        await self.session.commit()
        store = await self.vs_repo.get(store_id)  # type: ignore[assignment]
        return _to_vector_store_object(store)

    async def delete_store(self, store_id: str) -> DeleteResponse | None:
        deleted = await self.vs_repo.delete(store_id)
        if not deleted:
            return None
        await self.session.commit()
        return DeleteResponse(id=store_id, object="vector_store.deleted", deleted=True)

    # ── Files ────────────────────────────────────────────────────────

    async def attach_file(
        self, store_id: str, request: CreateVectorStoreFileRequest
    ) -> VectorStoreFileObject | None:
        store = await self.vs_repo.get(store_id)
        if store is None:
            return None
        chunking = _derive_chunking_strategy(
            store.chunking_strategy,
            store.chunk_size_tokens,
            store.chunk_overlap_tokens,
        )
        # Idempotent re-attach
        existing = await self.vf_repo.get_by_store_and_document(
            store_id, request.file_id
        )
        if existing is not None:
            if existing.status in ("cancelled", "failed"):
                # Reset to pending for re-processing
                await self.vf_repo.update_status(
                    existing.id, status="pending", failure_reason=None, next_attempt_at=None
                )
                existing = await self.vf_repo.get(existing.id)
            await self.session.commit()
            return _to_vector_store_file_object(existing, store_id, chunking)  # type: ignore[arg-type]

        doc = await self.doc_repo.get_document(request.file_id)
        if doc is None:
            return None
        attributes_json = json.dumps(request.attributes or {})
        initial_status = "completed" if await self._is_document_ready(doc) else "pending"
        vf = VectorStoreFileModel(
            id=_new_file_id(),
            vector_store_id=store_id,
            source_document_id=request.file_id,
            status=initial_status,
            bytes=doc.bytes or 0,
            attributes_json=attributes_json,
            completed_at=_utcnow() if initial_status == "completed" else None,
        )
        await self.vf_repo.create(vf)
        if initial_status == "completed":
            await self.doc_repo.update_chunks_vector_store_id(request.file_id, store_id)
            from apps.api.dependencies import qdrant_store
            if qdrant_store is not None:
                await qdrant_store.update_chunks_vector_store_id(request.file_id, store_id)
        await self.vf_repo.update_store_file_counts(store_id)
        # Bump last_active_at
        await self.vs_repo.update(store_id, last_active_at=_utcnow())
        await self.session.commit()
        return _to_vector_store_file_object(vf, store_id, chunking)

    async def get_file(
        self, store_id: str, file_id: str
    ) -> VectorStoreFileObject | None:
        vf = await self.vf_repo.get(file_id)
        if vf is None or vf.vector_store_id != store_id:
            return None
        store = await self.vs_repo.get(store_id)
        chunking = _derive_chunking_strategy(
            store.chunking_strategy if store else "auto",
            store.chunk_size_tokens if store else None,
            store.chunk_overlap_tokens if store else None,
        )
        return _to_vector_store_file_object(vf, store_id, chunking)

    async def list_files(
        self,
        store_id: str,
        *,
        limit: int = 20,
        after_id: str | None = None,
        before_id: str | None = None,
        status_filter: str | None = None,
        order: str = "desc",
    ) -> ListVectorStoreFilesResponse:
        store = await self.vs_repo.get(store_id)
        if store is None:
            return ListVectorStoreFilesResponse(data=[], has_more=False)

        chunking = _derive_chunking_strategy(
            store.chunking_strategy,
            store.chunk_size_tokens,
            store.chunk_overlap_tokens,
        )

        # Map OpenAI filter to internal status. "in_progress" means
        # all non-terminal statuses.
        internal_status: str | None = None
        if status_filter:
            if status_filter == "in_progress":
                # Use a SQL IN check via multiple calls — simpler is to query for non-terminal
                internal_status = None  # we'll filter post-query below
            elif status_filter == "completed":
                internal_status = "completed"
            elif status_filter == "cancelled":
                internal_status = "cancelled"
            elif status_filter == "failed":
                internal_status = "failed"

        if status_filter == "in_progress":
            # Fetch all and filter in Python for non-terminal statuses
            files = await self.vf_repo.list_by_store(
                store_id,
                limit=limit,
                after_id=after_id,
                before_id=before_id,
                status_filter=None,
                order=order,
            )
            # has_more detection first
            has_more_raw = len(files) > limit
            files = files[:limit]
            files = [f for f in files if f.status not in ("completed", "cancelled", "failed")]
            data = [_to_vector_store_file_object(f, store_id, chunking) for f in files]
            has_more = has_more_raw or len(data) > limit
        else:
            files = await self.vf_repo.list_by_store(
                store_id,
                limit=limit,
                after_id=after_id,
                before_id=before_id,
                status_filter=internal_status,
                order=order,
            )
            has_more = len(files) > limit
            files = files[:limit]
            data = [_to_vector_store_file_object(f, store_id, chunking) for f in files]

        return ListVectorStoreFilesResponse(
            data=data,
            has_more=has_more,
            first_id=data[0].id if data else None,
            last_id=data[-1].id if data else None,
        )

    async def update_file_attributes(
        self,
        store_id: str,
        file_id: str,
        request: UpdateVectorStoreFileRequest,
    ) -> VectorStoreFileObject | None:
        vf = await self.vf_repo.get(file_id)
        if vf is None or vf.vector_store_id != store_id:
            return None
        await self.vf_repo.update_attributes(file_id, request.attributes)
        await self.session.commit()
        vf = await self.vf_repo.get(file_id)  # type: ignore[assignment]
        store = await self.vs_repo.get(store_id)
        chunking = _derive_chunking_strategy(
            store.chunking_strategy if store else "auto",
            store.chunk_size_tokens if store else None,
            store.chunk_overlap_tokens if store else None,
        )
        return _to_vector_store_file_object(vf, store_id, chunking)  # type: ignore[arg-type]

    async def get_file_content(
        self,
        store_id: str,
        file_id: str,
    ) -> FileContentResponse | None:
        vf = await self.vf_repo.get(file_id)
        if vf is None or vf.vector_store_id != store_id:
            return None
        doc = await self.doc_repo.get_document(vf.source_document_id)
        doc_title = doc.title if doc else "unknown"
        # Get the chunks that belong to this (store, document) pair
        chunks = await self.doc_repo.get_chunks_by_vector_store_file(
            store_id, vf.source_document_id
        )
        attributes: dict = {}
        try:
            attributes = json.loads(vf.attributes_json or "{}")
        except (json.JSONDecodeError, TypeError):
            attributes = {}
        # Map to FileContentItem: one item per chunk
        data: list[FileContentItem] = []
        for c in chunks:
            data.append(FileContentItem(type="text", text=c.content))
        return FileContentResponse(
            data=data,
            has_more=False,
            next_page=None,
            file_id=file_id,
            filename=doc_title,
            attributes=attributes,
        )

    async def delete_file(
        self, store_id: str, file_id: str
    ) -> DeleteResponse | None:
        vf = await self.vf_repo.get(file_id)
        if vf is None or vf.vector_store_id != store_id:
            return None
        # Remove the file row entirely, then recompute counts
        await self.vf_repo.delete_by_file_id(file_id)
        await self.vf_repo.update_store_file_counts(store_id)
        await self.session.commit()
        return DeleteResponse(
            id=file_id, object="vector_store.file.deleted", deleted=True
        )

    # ── File Batches ─────────────────────────────────────────────────

    async def create_batch(
        self, store_id: str, request: CreateVectorStoreFileBatchRequest
    ) -> VectorStoreFileBatchObject | None:
        store = await self.vs_repo.get(store_id)
        if store is None:
            return None

        # Resolve the per-file list. file_ids and files are mutually exclusive.
        per_file: list[PerFileConfig] = []
        if request.files is not None and request.file_ids is not None:
            # Spec: mutually exclusive. Pick the more specific one (files).
            per_file = list(request.files)
        elif request.files is not None:
            per_file = list(request.files)
        elif request.file_ids is not None:
            per_file = [
                PerFileConfig(
                    file_id=fid,
                    attributes=request.attributes or {},
                    chunking_strategy=request.chunking_strategy,
                )
                for fid in request.file_ids
            ]
        # else: empty batch (the spec allows POST /file_batches with body '{}')

        default_chunking = _derive_chunking_strategy(
            store.chunking_strategy,
            store.chunk_size_tokens,
            store.chunk_overlap_tokens,
        )

        batch = VectorStoreFileBatchModel(
            id=_new_batch_id(),
            vector_store_id=store_id,
            status="in_progress",
            attributes_json=json.dumps(request.attributes or {}),
        )
        await self.vfb_repo.create(batch)

        for cfg in per_file:
            # Idempotent: if a file row already exists for (store, doc), reuse it.
            existing = await self.vf_repo.get_by_store_and_document(
                store_id, cfg.file_id
            )
            if existing is not None:
                await self.vf_repo.update_attributes(
                    existing.id, cfg.attributes or {}
                )
                # Re-attach to this batch (allow re-use across batches).
                await self.session.execute(
                    sa_update(VectorStoreFileModel)
                    .where(VectorStoreFileModel.id == existing.id)
                    .values(batch_id=batch.id)
                )
                continue
            doc = await self.doc_repo.get_document(cfg.file_id)
            if doc is None:
                logger.warning(
                    "vs_batch_attach_missing_document",
                    file_id=cfg.file_id,
                    batch_id=batch.id,
                    store_id=store_id,
                )
                continue
            vf = VectorStoreFileModel(
                id=_new_file_id(),
                vector_store_id=store_id,
                source_document_id=cfg.file_id,
                status="pending",
                bytes=doc.bytes or 0,
                attributes_json=json.dumps(cfg.attributes or {}),
                batch_id=batch.id,
            )
            await self.vf_repo.create(vf)

        await self.vf_repo.update_store_file_counts(store_id)
        counts = await self.vfb_repo.update_file_counts(batch.id)
        await self.vs_repo.update(store_id, last_active_at=_utcnow())
        await self.session.commit()

        return _to_vector_store_file_batch_object(batch, counts)

    async def get_batch(
        self, store_id: str, batch_id: str
    ) -> VectorStoreFileBatchObject | None:
        batch = await self.vfb_repo.get(batch_id)
        if batch is None or batch.vector_store_id != store_id:
            return None
        counts = json.loads(batch.file_counts_json or "{}")
        return _to_vector_store_file_batch_object(batch, counts)

    async def list_batch_files(
        self,
        store_id: str,
        batch_id: str,
        *,
        limit: int = 20,
        after_id: str | None = None,
        before_id: str | None = None,
        status_filter: str | None = None,
        order: str = "desc",
    ) -> ListVectorStoreFilesResponse | None:
        batch = await self.vfb_repo.get(batch_id)
        if batch is None or batch.vector_store_id != store_id:
            return None

        store = await self.vs_repo.get(store_id)
        chunking = _derive_chunking_strategy(
            store.chunking_strategy if store else "auto",
            store.chunk_size_tokens if store else None,
            store.chunk_overlap_tokens if store else None,
        )

        if status_filter == "in_progress":
            files = await self.vfb_repo.list_files_in_batch(
                batch_id,
                limit=limit,
                after_id=after_id,
                before_id=before_id,
                status_filter=None,
                order=order,
            )
            has_more_raw = len(files) > limit
            files = files[:limit]
            files = [
                f
                for f in files
                if f.status not in ("completed", "cancelled", "failed")
            ]
            data = [_to_vector_store_file_object(f, store_id, chunking) for f in files]
            has_more = has_more_raw or len(data) > limit
        else:
            internal_status = (
                status_filter
                if status_filter in ("completed", "cancelled", "failed")
                else None
            )
            files = await self.vfb_repo.list_files_in_batch(
                batch_id,
                limit=limit,
                after_id=after_id,
                before_id=before_id,
                status_filter=internal_status,
                order=order,
            )
            has_more = len(files) > limit
            files = files[:limit]
            data = [_to_vector_store_file_object(f, store_id, chunking) for f in files]

        return ListVectorStoreFilesResponse(
            data=data,
            has_more=has_more,
            first_id=data[0].id if data else None,
            last_id=data[-1].id if data else None,
        )

    async def cancel_batch(
        self, store_id: str, batch_id: str
    ) -> VectorStoreFileBatchObject | None:
        batch = await self.vfb_repo.get(batch_id)
        if batch is None or batch.vector_store_id != store_id:
            return None
        if batch.status == "cancelled":
            counts = json.loads(batch.file_counts_json or "{}")
            return _to_vector_store_file_batch_object(batch, counts)

        await self.vfb_repo.cancel(batch_id)
        # Update batch counts to reflect the cancellation wave. New claims will
        # skip files in this batch; in-flight files will complete normally.
        # We don't flip every file to 'cancelled' here — that would be eager and
        # contradicts the spec's "as soon as possible" semantics.
        counts = await self.vfb_repo.update_file_counts(batch_id)
        await self.session.commit()
        return _to_vector_store_file_batch_object(
            await self.vfb_repo.get(batch_id),  # type: ignore[arg-type]
            counts,
        )

    # ── Search ───────────────────────────────────────────────────────

    async def search(
        self, store_id: str, request: SearchRequest
    ) -> SearchResponse | None:
        store = await self.vs_repo.get(store_id)
        if store is None:
            return None
        if store.status == "expired":
            return None

        # Normalize query to list of strings
        queries = request.query if isinstance(request.query, list) else [request.query]
        if not queries:
            return SearchResponse(search_query=[])

        # Build filter predicate
        predicate = compile_predicate(request.filters)

        # Get embedder from app dependencies
        from apps.api.dependencies import embedder, qdrant_store

        if not all([qdrant_store, embedder]):
            return SearchResponse(search_query=queries)

        max_results = request.max_num_results

        # Embed each query and search
        # Allow Qdrant to return extra for filtering
        all_results: list[tuple[float, str, str, str, dict]] = []
        # (score, chunk_id, vector_store_file_id, filename, attributes)

        documents = await self.vf_repo.get_documents_by_store(store_id)

        # Pre-build doc_id → vector_store_file_id map (one per (store, document))
        from sqlalchemy import select

        from src.database.models import VectorStoreFileModel

        result = await self.session.execute(
            select(
                VectorStoreFileModel.id,
                VectorStoreFileModel.source_document_id,
            ).where(VectorStoreFileModel.vector_store_id == store_id)
        )
        doc_to_vf_id: dict[str, str] = {
            row.source_document_id: row.id for row in result.all()
        }

        for q in queries:
            embedding = await embedder.aembed_query(q)  # type: ignore[union-attr]
            dense = await qdrant_store.search_in_stores(  # type: ignore[union-attr]
                embedding,
                top_k=max_results,
                allowed_store_ids={store_id},
                predicate=predicate,
            )
            for r in dense:
                doc = documents.get(r.chunk.document_id)
                filename = doc.title if doc else "unknown"
                vf_id = doc_to_vf_id.get(r.chunk.document_id, "")
                all_results.append(
                    (r.score, r.chunk.id, vf_id, filename, r.chunk.attributes)
                )

        # De-dup by chunk_id, keep highest score
        best: dict[str, tuple[float, str, str, str, dict]] = {}
        for entry in all_results:
            cid = entry[1]
            if cid not in best or entry[0] > best[cid][0]:
                best[cid] = entry

        # Sort by score desc
        sorted_results = sorted(best.values(), key=lambda x: x[0], reverse=True)
        # Apply score threshold
        if request.ranking_options.score_threshold is not None:
            sorted_results = [
                r
                for r in sorted_results
                if r[0] >= request.ranking_options.score_threshold
            ]

        top = sorted_results[:max_results]
        has_more = len(sorted_results) > max_results

        # Build response items
        chunk_ids = [r[1] for r in top]
        chunk_contents: dict[str, str] = {}
        if chunk_ids:
            chunks = await self.doc_repo.get_chunks_by_ids(chunk_ids)
            chunk_contents = {c.id: c.content for c in chunks}

        items: list[SearchResultItem] = []
        for score, chunk_id, file_id, filename, attrs in top:
            text = chunk_contents.get(chunk_id, "")
            items.append(
                SearchResultItem(
                    file_id=file_id,
                    filename=filename,
                    score=score,
                    attributes=attrs,
                    content=[ContentBlock(type="text", text=text)],
                )
            )

        return SearchResponse(
            data=items,
            has_more=has_more,
            next_page=None,
            search_query=queries,
        )

    async def _get_file_id_for_chunk(
        self, chunk_id: str, document_id: str
    ) -> str:
        """Look up the vector_store_file_id for a chunk by document id.

        In our model, one document maps to one vector_store_file per store.
        We use a single cached lookup per search call.
        """
        # Note: real callers should batch this — we do it in the search method
        # by caching the (doc_id -> vf_id) map. The fallback here does the query.
        from sqlalchemy import select

        from src.database.models import VectorStoreFileModel

        result = await self.session.execute(
            select(VectorStoreFileModel.id)
            .where(VectorStoreFileModel.source_document_id == document_id)
            .limit(1)
        )
        row = result.first()
        return row[0] if row else ""
