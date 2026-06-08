"""OpenAI-compatible file endpoints backed by the local app stack."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import Response

from apps.api.schemas.files import FileDeletedResponse, FileListResponse, FileObject
from src.database.repositories import DocumentRepository
from src.database.session import async_session_factory
from src.ingestion.pipelines import IngestionPipeline
from src.observability.logging import get_logger
from src.storage.s3.client import S3Client
from src.shared.types import Document
from src.vector_stores.repository import VectorStoreFileRepository

router = APIRouter(prefix="/files", tags=["files"])
logger = get_logger()

ALLOWED_PURPOSES = {
    "assistants",
    "assistants_output",
    "batch",
    "batch_output",
    "fine-tune",
    "fine-tune-results",
    "vision",
    "user_data",
}


def _normalize_file_status(status: str | None) -> str:
    """Map internal processing states to the OpenAI-compatible file status set."""
    if status in ("error", "failed"):
        return "error"
    if status in ("processed", "completed"):
        return "processed"
    if status in ("processing", "in_progress"):
        return "processing"
    return "uploaded"


def _file_object_from_row(row, purpose: str, status: str = "uploaded") -> FileObject:
    metadata: dict = {}
    try:
        metadata = json.loads(row.metadata_json or "{}")
    except Exception:
        pass
    return FileObject(
        id=row.id,
        bytes=row.bytes or 0,
        created_at=int(row.created_at.timestamp()) if row.created_at else 0,
        filename=row.title or "file",
        purpose=purpose,
        status=_normalize_file_status(status),
        expires_at=None,
        status_details=None,
    )


async def _load_file_bytes(document) -> bytes | None:
    if document.s3_key:
        try:
            s3 = S3Client()
            return await s3.download(document.s3_key)
        except Exception:
            return None
    if document.content_text:
        return document.content_text.encode("utf-8")
    return None


@router.get("", response_model=FileListResponse)
async def list_files(
    after: str | None = Query(None),
    limit: int = Query(10000, ge=1, le=10000),
    order: str = Query("desc", pattern="^(asc|desc)$"),
    purpose: str | None = Query(None),
) -> FileListResponse:
    async with async_session_factory() as session:
        repo = DocumentRepository(session)
        rows = await repo.list_documents(
            limit=limit, after_id=after, order=order, purpose=purpose
        )

    has_more = len(rows) > limit
    rows = rows[:limit]
    data = []
    for row in rows:
        meta = {}
        try:
            meta = json.loads(row.metadata_json or "{}")
        except Exception:
            meta = {}
        data.append(
            _file_object_from_row(
                row,
                meta.get("purpose", "assistants"),
                meta.get("processing_status", "uploaded"),
            )
        )

    return FileListResponse(
        data=data,
        first_id=data[0].id if data else None,
        last_id=data[-1].id if data else None,
        has_more=has_more,
    )


@router.post("", response_model=FileObject)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    purpose: str = Form("assistants"),
) -> FileObject:
    if purpose not in ALLOWED_PURPOSES:
        raise HTTPException(status_code=400, detail=f"Unsupported purpose '{purpose}'")

    file_bytes = await file.read()
    suffix = Path(file.filename or "upload.bin").suffix or ".bin"
    file_id = f"file-{uuid4().hex}"

    try:
        s3 = S3Client()
        async with async_session_factory() as session:
            repo = DocumentRepository(session)
            document = Document(
                id=file_id,
                title=file.filename or "Untitled",
                source_path=file.filename or "upload.bin",
                source_type=suffix.lstrip(".") or "bin",
                metadata={
                    "file_id": file_id,
                    "purpose": purpose,
                    "filename": file.filename or "file",
                    "content_type": file.content_type or "application/octet-stream",
                    "processing_status": "uploaded",
                },
            )
            saved = await repo.create_document(document)
            s3_key = f"files/{saved.id}/{file.filename or 'upload.bin'}"
            await s3.upload(
                s3_key,
                file_bytes,
                content_type=file.content_type or "application/octet-stream",
            )
            await repo.update_document_metadata(
                saved.id,
                metadata_json=json.dumps(
                    {
                        **document.metadata,
                        "purpose": purpose,
                        "filename": file.filename or "file",
                        "content_type": file.content_type or "application/octet-stream",
                        "s3_key": s3_key,
                    }
                ),
                s3_key=s3_key,
                bytes=len(file_bytes),
            )
            await session.commit()

            background_tasks.add_task(_process_uploaded_file, saved.id)
            row = await repo.get_document_by_id(saved.id)
        if row is None:
            raise HTTPException(
                status_code=500, detail="File upload completed but record was not found"
            )
        return _file_object_from_row(row, purpose)
    except Exception as exc:
        logger.error(
            "file_upload_failed",
            error=str(exc),
            error_type=type(exc).__name__,
            exc_info=True,
        )
        raise HTTPException(status_code=502, detail="Failed to upload file.") from exc


_FILE_MAX_RETRIES = 3
_FILE_RETRY_DELAY_BASE = 5  # seconds, exponential backoff


async def _set_file_status(document_id: str, status: str) -> None:
    """Update the processing_status on the document metadata."""
    try:
        async with async_session_factory() as session:
            repo = DocumentRepository(session)
            row = await repo.get_document_by_id(document_id)
            if row is None:
                return
            meta = json.loads(row.metadata_json or "{}")
            meta["processing_status"] = status
            await repo.update_document_metadata(
                document_id, metadata_json=json.dumps(meta)
            )
            await session.commit()
    except Exception:
        pass


async def _process_uploaded_file(document_id: str) -> None:
    import asyncio

    await _set_file_status(document_id, "processing")

    for attempt in range(1, _FILE_MAX_RETRIES + 1):
        try:
            async with async_session_factory() as session:
                pipeline = IngestionPipeline(session)
                doc = await pipeline.process_existing_document(document_id)
            if doc is not None:
                # ── Complete any vector store files waiting on this document ──
                await _complete_vs_files_for_document(document_id)
                return
            # process_existing_document returned None — missing s3_key or document
            logger.warning(
                "file_processing_returned_none",
                document_id=document_id,
                attempt=attempt,
            )
        except Exception as exc:
            logger.error(
                "async_file_processing_failed",
                document_id=document_id,
                attempt=attempt,
                error=str(exc),
                error_type=type(exc).__name__,
                exc_info=True,
            )

        if attempt < _FILE_MAX_RETRIES:
            delay = _FILE_RETRY_DELAY_BASE * (2 ** (attempt - 1))
            await asyncio.sleep(delay)

    # All retries exhausted
    await _set_file_status(document_id, "error")
    logger.error(
        "file_processing_permanently_failed",
        document_id=document_id,
        max_retries=_FILE_MAX_RETRIES,
    )


async def _complete_vs_files_for_document(document_id: str) -> None:
    """After a document finishes processing, complete all pending vector store
    files that reference it, tag Qdrant payloads, and recompute store counts.

    This is the single bridge between the upload pipeline and the vector store
    subsystem — it eliminates the need for the VS worker to re-process already-
    processed documents.
    """
    try:
        async with async_session_factory() as session:
            vf_repo = VectorStoreFileRepository(session)
            from src.database.repositories import DocumentRepository

            doc_repo = DocumentRepository(session)

            # 1. Mark all pending VF rows as completed
            affected_store_ids = await vf_repo.complete_pending_for_document(
                document_id
            )
            await session.commit()

            if not affected_store_ids:
                return

            # 2. Tag DB chunk rows + Qdrant payloads for each store
            from apps.api.dependencies import qdrant_store

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
    except Exception as exc:
        logger.error(
            "vs_files_completion_failed",
            document_id=document_id,
            error=str(exc),
            error_type=type(exc).__name__,
            exc_info=True,
        )


@router.get("/{file_id}", response_model=FileObject)
async def retrieve_file(file_id: str) -> FileObject:
    async with async_session_factory() as session:
        repo = DocumentRepository(session)
        row = await repo.get_document_by_id(file_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"File '{file_id}' not found")
    try:
        meta = json.loads(row.metadata_json or "{}")
    except Exception:
        meta = {}
    return _file_object_from_row(
        row,
        meta.get("purpose", "assistants"),
        meta.get("processing_status", "uploaded"),
    )


@router.delete("/{file_id}", response_model=FileDeletedResponse)
async def delete_file(file_id: str) -> FileDeletedResponse:
    async with async_session_factory() as session:
        repo = DocumentRepository(session)
        vf_repo = VectorStoreFileRepository(session)
        row = await repo.get_document_by_id(file_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"File '{file_id}' not found")
        if row.s3_key:
            try:
                s3 = S3Client()
                await s3.delete(row.s3_key)
            except Exception:
                pass
        # Collect affected stores, then delete VF rows explicitly, then the
        # document, and recompute counts after the VF rows are gone.
        attached = await vf_repo.get_by_document_id(file_id)
        affected_store_ids = {vf.vector_store_id for vf in (attached or [])}
        if attached:
            await vf_repo.delete_by_document_id(file_id)
        await repo.delete_chunks_by_document(file_id)
        await repo.delete_document(file_id)
        await session.commit()
        for store_id in affected_store_ids:
            await vf_repo.update_store_file_counts(store_id)
        if affected_store_ids:
            await session.commit()
    return FileDeletedResponse(id=file_id, deleted=True, object="file")


@router.get("/{file_id}/content")
async def retrieve_file_content(file_id: str) -> Response:
    async with async_session_factory() as session:
        repo = DocumentRepository(session)
        row = await repo.get_document_by_id(file_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"File '{file_id}' not found")
    data = await _load_file_bytes(row)
    if data is None:
        raise HTTPException(
            status_code=404, detail=f"Content for file '{file_id}' not available"
        )
    return Response(content=data, media_type="application/octet-stream")
