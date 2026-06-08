"""OpenAI-compatible vector store schemas.

Mirror the public OpenAI Vector Stores API shape so callers can use the same
JSON for both the official SDK and this service.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Chunking strategies
# ---------------------------------------------------------------------------


class StaticChunkingStrategyConfig(BaseModel):
    max_chunk_size_tokens: int = Field(default=800, ge=100, le=4096)
    chunk_overlap_tokens: int = Field(default=400, ge=0, le=2048)


class StaticChunkingStrategy(BaseModel):
    type: Literal["static"] = "static"
    static: StaticChunkingStrategyConfig


class AutoChunkingStrategy(BaseModel):
    type: Literal["auto"] = "auto"


# OpenAI accepts "other" too; we map anything we don't recognise to auto.
class OtherChunkingStrategy(BaseModel):
    type: Literal["other"] = "other"


ChunkingStrategy = Annotated[
    Union[StaticChunkingStrategy, AutoChunkingStrategy, OtherChunkingStrategy],
    Field(discriminator="type"),
]


def chunking_strategy_to_internal(
    strategy: ChunkingStrategy | None,
) -> tuple[str, int | None, int | None]:
    """Convert an OpenAI chunking strategy to the (name, size, overlap) tuple
    used internally by the chunker and persisted on VectorStoreModel."""
    if strategy is None:
        return ("auto", None, None)
    if isinstance(strategy, StaticChunkingStrategy):
        cfg = strategy.static
        return ("static", cfg.max_chunk_size_tokens, cfg.chunk_overlap_tokens)
    return ("auto", None, None)


# ---------------------------------------------------------------------------
# Expiration
# ---------------------------------------------------------------------------


class ExpiresAfter(BaseModel):
    anchor: Literal["last_active_at"] = "last_active_at"
    days: int = Field(ge=1, le=365)


# ---------------------------------------------------------------------------
# File counts / status
# ---------------------------------------------------------------------------


class FileCounts(BaseModel):
    in_progress: int = 0
    completed: int = 0
    cancelled: int = 0
    failed: int = 0
    total: int = 0


class VectorStoreFileStatus(BaseModel):
    """Per-file processing status. Mirrors OpenAI's lifecycle."""

    status: Literal[
        "pending",
        "processing",
        "chunking",
        "embedding",
        "indexing",
        "completed",
        "failed",
        "cancelled",
    ] = "pending"
    failure_reason: str | None = None


# ---------------------------------------------------------------------------
# VectorStore object
# ---------------------------------------------------------------------------


class VectorStoreObject(BaseModel):
    id: str
    object: Literal["vector_store"] = "vector_store"
    created_at: int  # epoch seconds
    name: str = ""
    bytes: int = 0
    status: Literal["expired", "in_progress", "completed"] = "in_progress"
    file_counts: FileCounts = Field(default_factory=FileCounts)
    last_active_at: int = 0
    metadata: dict[str, str] = Field(default_factory=dict)
    expires_after: ExpiresAfter | None = None
    expires_at: int | None = None


# ---------------------------------------------------------------------------
# File object (within a vector store)
# ---------------------------------------------------------------------------


class LastError(BaseModel):
    code: Literal["server_error", "unsupported_file", "invalid_file"] = "server_error"
    message: str


class VectorStoreFileObject(BaseModel):
    id: str  # file-<uuid>
    object: Literal["vector_store.file"] = "vector_store.file"
    created_at: int
    vector_store_id: str
    status: Literal[
        "in_progress",
        "completed",
        "cancelled",
        "failed",
    ] = "in_progress"
    last_error: LastError | None = None
    bytes: int = 0
    usage_bytes: int = 0
    chunking_strategy: ChunkingStrategy | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class UpdateVectorStoreFileRequest(BaseModel):
    attributes: dict[str, Any] = Field(default_factory=dict)


# File filter for list endpoint
class VectorStoreFileStatusFilter(StrEnum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# File content response
class FileContentItem(BaseModel):
    type: str
    text: str | None = None


class FileContentResponse(BaseModel):
    object: Literal["vector_store.file_content.page"] = "vector_store.file_content.page"
    data: list[FileContentItem] = Field(default_factory=list)
    has_more: bool = False
    next_page: str | None = None
    file_id: str = ""
    filename: str = ""
    attributes: dict[str, Any] = Field(default_factory=dict)


# List filter for files
class ListVectorStoreFilesQuery(BaseModel):
    after: str | None = None
    before: str | None = None
    filter: VectorStoreFileStatusFilter | None = None
    limit: int = Field(default=20, ge=1, le=100)
    order: Literal["asc", "desc"] = "desc"


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------


class CreateVectorStoreRequest(BaseModel):
    name: str | None = None
    file_ids: list[str] = Field(default_factory=list)
    chunking_strategy: ChunkingStrategy | None = None
    expires_after: ExpiresAfter | None = None
    metadata: dict[str, str] | None = None


class UpdateVectorStoreRequest(BaseModel):
    name: str | None = None
    expires_after: ExpiresAfter | None = None
    metadata: dict[str, str] | None = None


class CreateVectorStoreFileRequest(BaseModel):
    file_id: str
    chunking_strategy: ChunkingStrategy | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class ComparisonFilter(BaseModel):
    type: Literal["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin"]
    key: str
    value: Any


class AndFilter(BaseModel):
    type: Literal["and"] = "and"
    filters: list["CompoundFilter"]


class OrFilter(BaseModel):
    type: Literal["or"] = "or"
    filters: list["CompoundFilter"]


CompoundFilter = Annotated[
    Union[ComparisonFilter, AndFilter, OrFilter],
    Field(discriminator="type"),
]
AndFilter.model_rebuild()
OrFilter.model_rebuild()


class RankingOptions(BaseModel):
    ranker: Literal["none", "auto", "default-2024-11-15"] = "none"
    score_threshold: float | None = None


class SearchRequest(BaseModel):
    query: str | list[str] = Field(..., min_length=1)
    filters: CompoundFilter | None = None
    max_num_results: int = Field(default=10, ge=1, le=100)
    ranking_options: RankingOptions = Field(default_factory=RankingOptions)
    rewrite_query: bool = False


class ContentBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class SearchResultItem(BaseModel):
    file_id: str
    filename: str
    score: float
    attributes: dict[str, Any] = Field(default_factory=dict)
    content: list[ContentBlock]


class SearchResponse(BaseModel):
    object: Literal["vector_store.search_results.page"] = (
        "vector_store.search_results.page"
    )
    data: list[SearchResultItem] = Field(default_factory=list)
    has_more: bool = False
    next_page: str | None = None
    search_query: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# List responses
# ---------------------------------------------------------------------------


class ListVectorStoresResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[VectorStoreObject] = Field(default_factory=list)
    has_more: bool = False
    first_id: str | None = None
    last_id: str | None = None


class ListVectorStoreFilesResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[VectorStoreFileObject] = Field(default_factory=list)
    has_more: bool = False
    first_id: str | None = None
    last_id: str | None = None


class DeleteResponse(BaseModel):
    id: str
    object: Literal["vector_store.deleted", "vector_store.file.deleted"] = (
        "vector_store.deleted"
    )
    deleted: bool = True


# ---------------------------------------------------------------------------
# File Batches
# ---------------------------------------------------------------------------


class VectorStoreFileBatchFileCounts(BaseModel):
    in_progress: int = 0
    completed: int = 0
    cancelled: int = 0
    failed: int = 0
    total: int = 0


class VectorStoreFileBatchObject(BaseModel):
    id: str  # vsfb_<uuid>
    object: Literal["vector_store.files_batch"] = "vector_store.files_batch"
    created_at: int  # epoch seconds
    vector_store_id: str
    status: Literal["in_progress", "completed", "cancelled", "failed"] = "in_progress"
    file_counts: VectorStoreFileBatchFileCounts = Field(
        default_factory=VectorStoreFileBatchFileCounts
    )


class PerFileConfig(BaseModel):
    file_id: str
    attributes: dict[str, Any] = Field(default_factory=dict)
    chunking_strategy: ChunkingStrategy | None = None


class CreateVectorStoreFileBatchRequest(BaseModel):
    file_ids: list[str] | None = None
    files: list[PerFileConfig] | None = None
    attributes: dict[str, Any] | None = None
    chunking_strategy: ChunkingStrategy | None = None


class VectorStoreFileBatchStatusFilter(StrEnum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
