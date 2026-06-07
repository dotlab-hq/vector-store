"""OpenAI-compatible vector stores API endpoints.

Mirrors the public OpenAI Vector Stores REST API:

    POST   /vector_stores                                create store
    GET    /vector_stores                                list stores
    GET    /vector_stores/{id}                           get store
    POST   /vector_stores/{id}                           modify store
    DELETE /vector_stores/{id}                           delete store
    POST   /vector_stores/{id}/files                     attach file
    GET    /vector_stores/{id}/files                     list files
    GET    /vector_stores/{id}/files/{fid}               get file
    POST   /vector_stores/{id}/files/{fid}               update file attributes
    DELETE /vector_stores/{id}/files/{fid}               remove file
    GET    /vector_stores/{id}/files/{fid}/content       get file content
    POST   /vector_stores/{id}/search                    search
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import APIRouter, HTTPException, Query

from apps.api.schemas.vector_stores import (
    CreateVectorStoreFileBatchRequest,
    CreateVectorStoreFileRequest,
    CreateVectorStoreRequest,
    DeleteResponse,
    FileContentResponse,
    ListVectorStoreFilesResponse,
    ListVectorStoresResponse,
    SearchRequest,
    SearchResponse,
    UpdateVectorStoreFileRequest,
    UpdateVectorStoreRequest,
    VectorStoreFileBatchObject,
    VectorStoreFileBatchStatusFilter,
    VectorStoreFileObject,
    VectorStoreFileStatusFilter,
    VectorStoreObject,
)
from src.database.session import async_session_factory
from src.observability.logging import get_logger
from src.vector_stores.service import VectorStoreService

router = APIRouter(prefix="/vector_stores", tags=["vector_stores"])
logger = get_logger()


@asynccontextmanager
async def _service_in_session():
    async with async_session_factory() as session:
        yield VectorStoreService(session)


# ── Vector Stores ────────────────────────────────────────────────────


@router.post("", response_model=VectorStoreObject)
async def create_vector_store(
    request: CreateVectorStoreRequest,
) -> VectorStoreObject:
    async with _service_in_session() as svc:
        return await svc.create_store(request)


@router.get("", response_model=ListVectorStoresResponse)
async def list_vector_stores(
    limit: int = Query(20, ge=1, le=100),
    after: str | None = Query(None),
) -> ListVectorStoresResponse:
    async with _service_in_session() as svc:
        return await svc.list_stores(limit=limit, after_id=after)


@router.get("/{vector_store_id}", response_model=VectorStoreObject)
async def get_vector_store(vector_store_id: str) -> VectorStoreObject:
    async with _service_in_session() as svc:
        result = await svc.get_store(vector_store_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Vector store '{vector_store_id}' not found")
    return result


@router.post("/{vector_store_id}", response_model=VectorStoreObject)
async def modify_vector_store(
    vector_store_id: str,
    request: UpdateVectorStoreRequest,
) -> VectorStoreObject:
    async with _service_in_session() as svc:
        result = await svc.update_store(vector_store_id, request)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Vector store '{vector_store_id}' not found")
    return result


@router.delete("/{vector_store_id}", response_model=DeleteResponse)
async def delete_vector_store(vector_store_id: str) -> DeleteResponse:
    async with _service_in_session() as svc:
        result = await svc.delete_store(vector_store_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Vector store '{vector_store_id}' not found")
    return result


# ── Files ───────────────────────────────────────────────────────────


@router.post(
    "/{vector_store_id}/files", response_model=VectorStoreFileObject
)
async def attach_file(
    vector_store_id: str,
    request: CreateVectorStoreFileRequest,
) -> VectorStoreFileObject:
    async with _service_in_session() as svc:
        result = await svc.attach_file(vector_store_id, request)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Vector store '{vector_store_id}' or file '{request.file_id}' not found",
        )
    return result


@router.get(
    "/{vector_store_id}/files", response_model=ListVectorStoreFilesResponse
)
async def list_files(
    vector_store_id: str,
    limit: int = Query(20, ge=1, le=100),
    after: str | None = Query(None),
    before: str | None = Query(None),
    filter: VectorStoreFileStatusFilter | None = Query(None, alias="filter"),
    order: str = Query("desc"),
) -> ListVectorStoreFilesResponse:
    if order not in ("asc", "desc"):
        raise HTTPException(
            status_code=400, detail="order must be 'asc' or 'desc'"
        )
    async with _service_in_session() as svc:
        return await svc.list_files(
            vector_store_id,
            limit=limit,
            after_id=after,
            before_id=before,
            status_filter=filter.value if filter else None,
            order=order,
        )


@router.get(
    "/{vector_store_id}/files/{file_id}", response_model=VectorStoreFileObject
)
async def get_file(vector_store_id: str, file_id: str) -> VectorStoreFileObject:
    async with _service_in_session() as svc:
        result = await svc.get_file(vector_store_id, file_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"File '{file_id}' not found in vector store '{vector_store_id}'",
        )
    return result


@router.post(
    "/{vector_store_id}/files/{file_id}", response_model=VectorStoreFileObject
)
async def update_file_attributes(
    vector_store_id: str,
    file_id: str,
    request: UpdateVectorStoreFileRequest,
) -> VectorStoreFileObject:
    async with _service_in_session() as svc:
        result = await svc.update_file_attributes(vector_store_id, file_id, request)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"File '{file_id}' not found in vector store '{vector_store_id}'",
        )
    return result


@router.delete(
    "/{vector_store_id}/files/{file_id}", response_model=DeleteResponse
)
async def remove_file(vector_store_id: str, file_id: str) -> DeleteResponse:
    async with _service_in_session() as svc:
        result = await svc.delete_file(vector_store_id, file_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"File '{file_id}' not found in vector store '{vector_store_id}'",
        )
    return result


@router.get(
    "/{vector_store_id}/files/{file_id}/content",
    response_model=FileContentResponse,
)
async def get_file_content(
    vector_store_id: str, file_id: str
) -> FileContentResponse:
    async with _service_in_session() as svc:
        result = await svc.get_file_content(vector_store_id, file_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"File '{file_id}' not found in vector store '{vector_store_id}'",
        )
    return result


# ── File Batches ────────────────────────────────────────────────────


@router.post(
    "/{vector_store_id}/file_batches",
    response_model=VectorStoreFileBatchObject,
)
async def create_file_batch(
    vector_store_id: str,
    request: CreateVectorStoreFileBatchRequest,
) -> VectorStoreFileBatchObject:
    async with _service_in_session() as svc:
        result = await svc.create_batch(vector_store_id, request)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Vector store '{vector_store_id}' not found",
        )
    return result


@router.get(
    "/{vector_store_id}/file_batches/{batch_id}",
    response_model=VectorStoreFileBatchObject,
)
async def get_file_batch(
    vector_store_id: str, batch_id: str
) -> VectorStoreFileBatchObject:
    async with _service_in_session() as svc:
        result = await svc.get_batch(vector_store_id, batch_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Batch '{batch_id}' not found in vector store '{vector_store_id}'",
        )
    return result


@router.get(
    "/{vector_store_id}/file_batches/{batch_id}/files",
    response_model=ListVectorStoreFilesResponse,
)
async def list_batch_files(
    vector_store_id: str,
    batch_id: str,
    limit: int = Query(20, ge=1, le=100),
    after: str | None = Query(None),
    before: str | None = Query(None),
    filter: VectorStoreFileBatchStatusFilter | None = Query(None, alias="filter"),
    order: str = Query("desc"),
) -> ListVectorStoreFilesResponse:
    if order not in ("asc", "desc"):
        raise HTTPException(
            status_code=400, detail="order must be 'asc' or 'desc'"
        )
    async with _service_in_session() as svc:
        result = await svc.list_batch_files(
            vector_store_id,
            batch_id,
            limit=limit,
            after_id=after,
            before_id=before,
            status_filter=filter.value if filter else None,
            order=order,
        )
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Batch '{batch_id}' not found in vector store '{vector_store_id}'",
        )
    return result


@router.post(
    "/{vector_store_id}/file_batches/{batch_id}/cancel",
    response_model=VectorStoreFileBatchObject,
)
async def cancel_file_batch(
    vector_store_id: str, batch_id: str
) -> VectorStoreFileBatchObject:
    async with _service_in_session() as svc:
        result = await svc.cancel_batch(vector_store_id, batch_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Batch '{batch_id}' not found in vector store '{vector_store_id}'",
        )
    return result


# ── Search ──────────────────────────────────────────────────────────


@router.post("/{vector_store_id}/search", response_model=SearchResponse)
async def search_vector_store(
    vector_store_id: str,
    request: SearchRequest,
) -> SearchResponse:
    async with _service_in_session() as svc:
        result = await svc.search(vector_store_id, request)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Vector store '{vector_store_id}' not found or expired",
        )
    return result
