"""OpenAI-compatible file endpoints backed by the local app stack.

File uploads create a DB record + a processing task row. Heavy work
(parse, chunk, embed, index) is deferred to the background worker via arq.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from uuid import uuid4

from fastapi import (
    APIRouter,
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
from src.observability.logging import get_logger
from src.shared.events import enqueue_task
from src.shared.types import Document

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


def _sanitize_filename(filename: str) -> str:
    """Remove path traversal sequences from a filename."""
    name = Path(filename).name
    return re.sub(r"[^A-Za-z0-9.\-_]", "_", name) or "upload.bin"


def _normalize_file_status(status: str | None) -> str:
    """Map internal processing states to the OpenAI-compatible file status set."""
    if status in ("error", "failed"):
        return "error"
    if status in ("processed", "completed"):
        return "processed"
    if status in ("processing", "in_progress", "retrying"):
        return "processing"
    return "uploaded"


def _file_object_from_row(row, purpose: str, status: str = "uploaded") -> FileObject:
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
    file: UploadFile = File(...),
    purpose: str = Form("assistants"),
) -> FileObject:
    """Upload a file — stores in S3, creates DB record + task row."""
    if purpose not in ALLOWED_PURPOSES:
        raise HTTPException(status_code=400, detail=f"Unsupported purpose '{purpose}'")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    suffix = Path(file.filename or "upload.bin").suffix or ".bin"
    file_id = f"file-{uuid4().hex}"

    from src.config import settings
    from src.storage.s3.client import S3Client

    s3_key = None
    s3 = None
    try:
        if settings.s3_access_key:
            s3 = S3Client()
            safe_name = _sanitize_filename(file.filename or "upload.bin")
            s3_key = f"files/{file_id}/{safe_name}"
            await s3.upload(
                s3_key,
                file_bytes,
                content_type=file.content_type or "application/octet-stream",
            )

        async with async_session_factory() as session:
            repo = DocumentRepository(session)

            document = Document(
                id=file_id,
                title=file.filename or "Untitled",
                source_path=file.filename or "upload.bin",
                source_type=suffix.lstrip(".") or "bin",
                metadata={
                    "purpose": purpose,
                    "filename": file.filename or "file",
                    "content_type": file.content_type or "application/octet-stream",
                    "processing_status": "uploaded",
                    "s3_key": s3_key,
                },
            )
            await repo.create_document(document)
            await repo.update_document_metadata(
                file_id,
                s3_key=s3_key,
                bytes=len(file_bytes),
            )
            await session.commit()

        # Enqueue background processing via arq
        await enqueue_task(
            "document.ingest",
            {"document_id": file_id, "s3_key": s3_key, "purpose": purpose},
        )

        async with async_session_factory() as session:
            repo = DocumentRepository(session)
            row = await repo.get_document_by_id(file_id)

        if row is None:
            raise HTTPException(
                status_code=500, detail="File upload completed but record was not found"
            )
        return _file_object_from_row(row, purpose)
    except HTTPException:
        raise
    except Exception as exc:
        # Clean up orphaned S3 object if DB operations failed
        if s3 is not None and s3_key is not None:
            try:
                await s3.delete(s3_key)
            except Exception:
                pass
        logger.error(
            "file_upload_failed",
            error=str(exc),
            error_type=type(exc).__name__,
            exc_info=True,
        )
        raise HTTPException(status_code=502, detail="Failed to upload file.") from exc


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
        from src.vector_stores.repository import VectorStoreFileRepository

        vf_repo = VectorStoreFileRepository(session)

        row = await repo.get_document_by_id(file_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"File '{file_id}' not found")

        if row.s3_key:
            try:
                from src.storage.s3.client import S3Client

                s3 = S3Client()
                await s3.delete(row.s3_key)
            except Exception:
                pass

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

    if row.s3_key:
        try:
            from src.storage.s3.client import S3Client

            s3 = S3Client()
            data = await s3.download(row.s3_key)
            return Response(content=data, media_type="application/octet-stream")
        except Exception:
            pass

    if row.content_text:
        return Response(
            content=row.content_text.encode("utf-8"),
            media_type="application/octet-stream",
        )

    raise HTTPException(
        status_code=404, detail=f"Content for file '{file_id}' not available"
    )
