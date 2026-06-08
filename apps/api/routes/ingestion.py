"""Ingestion endpoints — lightweight: create DB record + task row, return 202.

Heavy processing (parse, chunk, embed, index) is deferred to the background
worker via the processing_tasks queue.
"""

import re
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from apps.api.schemas import IngestResponse
from src.database.repositories import DocumentRepository, ProcessingTaskRepository
from src.database.session import async_session_factory
from src.observability.logging import get_logger
from src.shared.types import Document

def _sanitize_filename(filename: str) -> str:
    """Remove path traversal sequences from a filename."""
    name = Path(filename).name
    return re.sub(r"[^A-Za-z0-9.\-_]", "_", name) or "upload.bin"


router = APIRouter()
logger = get_logger()

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}
MAX_FILE_SIZE_MB = 50
MAX_TEXT_LENGTH = 500_000


@router.post("/ingest/text", response_model=IngestResponse)
async def ingest_text(
    text: str = Form(...), title: str = Form("Untitled")
) -> IngestResponse:
    """Ingest raw text content — defers chunking/indexing to background worker."""
    if len(text) > MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=413,
            detail=f"Text exceeds {MAX_TEXT_LENGTH} character limit",
        )
    document_id = uuid4().hex
    try:
        async with async_session_factory() as session:
            repo = DocumentRepository(session)
            task_repo = ProcessingTaskRepository(session)

            document = Document(
                id=document_id,
                title=title,
                source_path="",
                source_type="text",
                content_text=text,
            )
            await repo.create_document(document)
            await task_repo.create(
                task_type="document.ingest_text",
                payload={"document_id": document_id, "content_text": text, "title": title},
            )
            await session.commit()

        return IngestResponse(
            document_id=document_id,
            title=title,
            chunks_created=0,
            processing_status="pending",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "ingest_text_failed",
            error=str(exc),
            error_type=type(exc).__name__,
            exc_info=True,
        )
        raise HTTPException(status_code=502, detail="Failed to ingest text.") from exc


@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    title: str = Form(""),
) -> IngestResponse:
    """Upload and ingest a file (PDF, DOCX, TXT, MD, CSV) — defers processing."""
    suffix = Path(file.filename or "unknown").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    if file.size and file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413, detail=f"File exceeds {MAX_FILE_SIZE_MB}MB limit"
        )

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413, detail=f"File exceeds {MAX_FILE_SIZE_MB}MB limit"
        )
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    document_id = uuid4().hex
    doc_title = title or file.filename or "Untitled"

    s3_key = None
    s3 = None
    try:
        from src.config import settings
        from src.storage.s3.client import S3Client

        if settings.s3_access_key:
            s3 = S3Client()
            safe_name = _sanitize_filename(file.filename or "upload.bin")
            s3_key = f"files/{document_id}/{safe_name}"
            await s3.upload(
                s3_key,
                file_bytes,
                content_type=file.content_type or "application/octet-stream",
            )

        async with async_session_factory() as session:
            repo = DocumentRepository(session)
            task_repo = ProcessingTaskRepository(session)

            document = Document(
                id=document_id,
                title=doc_title,
                source_path=file.filename or "upload.bin",
                source_type=suffix.lstrip(".") or "bin",
                metadata={
                    "purpose": "assistants",
                    "filename": file.filename or "file",
                    "content_type": file.content_type or "application/octet-stream",
                    "processing_status": "uploaded",
                    "s3_key": s3_key,
                },
            )
            await repo.create_document(document)

            payload = {"document_id": document_id, "s3_key": s3_key}
            await task_repo.create(
                task_type="document.ingest",
                payload=payload,
            )
            await session.commit()

        return IngestResponse(
            document_id=document_id,
            title=doc_title,
            chunks_created=0,
            processing_status="pending",
        )
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
            "ingest_file_failed",
            error=str(exc),
            error_type=type(exc).__name__,
            exc_info=True,
        )
        raise HTTPException(status_code=502, detail="Failed to upload file.") from exc


@router.post("/ingest", response_model=IngestResponse)
async def ingest_file_or_text(
    file: UploadFile | None = File(None),
    text: str = Form(""),
    title: str = Form(""),
) -> IngestResponse:
    """Ingest a file upload OR raw text. Defers processing to background worker."""
    if file is not None and file.filename:
        return await ingest_file(file=file, title=title)
    elif text.strip():
        return await ingest_text(text=text, title=title or "Untitled")
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'file' or 'text' field",
        )


# --- Inspection endpoints ---


class ChunkInfo(BaseModel):
    id: str
    document_id: str
    parent_id: str | None
    content: str
    page_number: int | None
    section: str
    position: int


class DocumentChunksResponse(BaseModel):
    document_id: str
    chunks: list[ChunkInfo]


@router.get("/chunks/{document_id}", response_model=DocumentChunksResponse)
async def get_document_chunks(document_id: str) -> DocumentChunksResponse:
    """Get all chunks for a document from the database."""
    async with async_session_factory() as session:
        repo = DocumentRepository(session)
        chunks = await repo.get_chunks_by_document(document_id)

    if not chunks:
        raise HTTPException(
            status_code=404, detail=f"No chunks found for document '{document_id}'"
        )

    return DocumentChunksResponse(
        document_id=document_id,
        chunks=[
            ChunkInfo(
                id=c.id,
                document_id=c.document_id,
                parent_id=c.parent_id,
                content=c.content[:200] + "..." if len(c.content) > 200 else c.content,
                page_number=c.page_number,
                section=c.section,
                position=c.position,
            )
            for c in chunks
        ],
    )


class IndexStatsResponse(BaseModel):
    qdrant_count: int
    bm25_count: int


@router.get("/index/stats", response_model=IndexStatsResponse)
async def get_index_stats() -> IndexStatsResponse:
    """Check how many items are in the Qdrant and BM25 indexes."""
    from apps.api.dependencies import bm25_store, qdrant_store

    qdrant_count = await qdrant_store.count() if qdrant_store else 0
    bm25_count = await bm25_store.count() if bm25_store else 0

    return IndexStatsResponse(qdrant_count=qdrant_count, bm25_count=bm25_count)
