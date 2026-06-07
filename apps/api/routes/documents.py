"""
Document download route.

Generates a presigned URL for the original file stored in S3 and redirects the
client to it. The bucket is private — clients never see the raw object key,
only a short-lived signed URL.
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse

from src.config import settings
from src.database.repositories import DocumentRepository
from src.database.session import async_session_factory
from src.observability.logging import get_logger

logger = get_logger()
router = APIRouter()

# Presigned URLs are short-lived. The exact maximum depends on the S3-compatible
# provider; AWS caps at 7 days but 15 min is more than enough for a single click.
MAX_EXPIRES_IN = 3600  # 1 hour hard cap


@router.get("/documents/{document_id}/download")
async def download_document(
    document_id: str,
    expires_in: int = Query(None, ge=60, le=MAX_EXPIRES_IN),
) -> RedirectResponse:
    """Look up the document's S3 key, generate a presigned URL, and 307-redirect to it.

    This endpoint never streams the file through the API. It only mints a
    time-limited signed URL that the client uses to download directly from S3.

    Args:
        document_id: Internal document UUID.
        expires_in: Presigned URL lifetime in seconds (60-3600).

    Returns:
        307 Temporary Redirect to the presigned S3 URL.

    Raises:
        404: Document or S3 key not found.
        503: S3 not configured.
    """
    if not settings.s3_access_key or not settings.s3_secret_key:
        raise HTTPException(
            status_code=503,
            detail="Document storage is not available.",
        )

    # Default expiry from settings if caller didn't pass one
    if expires_in is None:
        expires_in = settings.s3_presign_expiry

    # 1. Look up the document and its s3_key in Postgres
    async with async_session_factory() as session:
        repo = DocumentRepository(session)
        document = await repo.get_document_by_id(document_id)

    if not document:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")

    s3_key = document.s3_key
    if not s3_key:
        # Fallback: some legacy docs may have s3_key in metadata JSON only.
        import json

        try:
            meta = json.loads(document.metadata_json or "{}")
            s3_key = meta.get("s3_key")
        except Exception:
            s3_key = None

    if not s3_key:
        raise HTTPException(
            status_code=404,
            detail=f"No S3 object associated with document '{document_id}'",
        )

    # 2. Mint a presigned URL via the existing S3Client adapter
    try:
        from src.storage.s3.client import S3Client

        s3 = S3Client()
        presigned_url = await s3.get_presigned_url(s3_key, expires_in=expires_in)
    except Exception as exc:
        logger.error(
            "presigned_url_generation_failed",
            document_id=document_id,
            s3_key=s3_key,
            error_type=type(exc).__name__,
            exc_info=True,
        )
        raise HTTPException(
            status_code=502,
            detail="Failed to generate presigned URL. Try again later.",
        ) from exc

    logger.info(
        "presigned_url_issued",
        document_id=document_id,
        s3_key=s3_key,
        expires_in=expires_in,
    )

    # 3. 307 preserves the method, but for a GET redirect either 302 or 307 works.
    #    302 is the more common choice for "GET me out of here to the resource."
    return RedirectResponse(url=presigned_url, status_code=302)
