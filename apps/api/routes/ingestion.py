import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from apps.api.schemas import IngestResponse
from src.database.repositories import DocumentRepository
from src.database.session import async_session_factory
from src.ingestion.pipelines import IngestionPipeline
from src.observability.logging import get_logger
from pydantic import BaseModel

router = APIRouter()
logger = get_logger()

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}
MAX_FILE_SIZE_MB = 50
MAX_TEXT_LENGTH = 500_000


@router.post("/ingest/text", response_model=IngestResponse)
async def ingest_text(text: str = Form(...), title: str = Form("Untitled")) -> IngestResponse:
    """Ingest raw text content."""
    async with async_session_factory() as session:
        pipeline = IngestionPipeline(session)
        document = await pipeline.ingest_text(text, title=title)

        repo = DocumentRepository(session)
        chunks = await repo.get_chunks_by_document(document.id)

    return IngestResponse(
        document_id=document.id,
        title=document.title,
        chunks_created=len(chunks),
    )


@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    title: str = Form(""),
) -> IngestResponse:
    """Upload and ingest a file (PDF, DOCX, TXT, MD, CSV)."""
    # Validate extension
    suffix = Path(file.filename or "unknown").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # Validate file size (max 50MB)
    if file.size and file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_FILE_SIZE_MB}MB limit")

    # Read file bytes and write to a temp file
    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_FILE_SIZE_MB}MB limit")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    try:
        async with async_session_factory() as session:
            pipeline = IngestionPipeline(session)
            document = await pipeline.ingest(
                tmp_path,
                title_override=title or file.filename or "Untitled",
            )

            repo = DocumentRepository(session)
            chunks = await repo.get_chunks_by_document(document.id)

        return IngestResponse(
            document_id=document.id,
            title=document.title,
            chunks_created=len(chunks),
        )
    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/ingest", response_model=IngestResponse)
async def ingest_file_or_text(
    file: UploadFile | None = File(None),
    text: str = Form(""),
    title: str = Form(""),
) -> IngestResponse:
    """Ingest a file upload OR raw text. Accepts multipart/form-data.

    Usage with curl:
        # Upload a file:
        curl -X POST http://localhost:8000/ingest -F "file=@document.pdf" -F "title=My Doc"

        # Or send text:
        curl -X POST http://localhost:8000/ingest -F "text=Your document content" -F "title=My Note"
    """
    if file is not None and file.filename:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{suffix}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            )

        # Validate file size before reading into memory
        if file.size and file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"File exceeds {MAX_FILE_SIZE_MB}MB limit")

        file_bytes = await file.read()
        if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"File exceeds {MAX_FILE_SIZE_MB}MB limit")

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = Path(tmp.name)

        try:
            async with async_session_factory() as session:
                pipeline = IngestionPipeline(session)
                document = await pipeline.ingest(
                    tmp_path,
                    title_override=title or file.filename or "Untitled",
                )
                repo = DocumentRepository(session)
                chunks = await repo.get_chunks_by_document(document.id)

            return IngestResponse(
                document_id=document.id,
                title=document.title,
                chunks_created=len(chunks),
            )
        finally:
            tmp_path.unlink(missing_ok=True)

    elif text.strip():
        if len(text) > MAX_TEXT_LENGTH:
            raise HTTPException(status_code=413, detail=f"Text exceeds {MAX_TEXT_LENGTH} character limit")
        async with async_session_factory() as session:
            pipeline = IngestionPipeline(session)
            document = await pipeline.ingest_text(text, title=title or "Untitled")
            repo = DocumentRepository(session)
            chunks = await repo.get_chunks_by_document(document.id)

        return IngestResponse(
            document_id=document.id,
            title=document.title,
            chunks_created=len(chunks),
        )

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
        raise HTTPException(status_code=404, detail=f"No chunks found for document '{document_id}'")

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
