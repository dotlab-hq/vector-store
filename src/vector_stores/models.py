"""Domain types and enums for vector stores."""

from __future__ import annotations

from enum import StrEnum

VECTOR_STORE_OBJECT = "vector_store"
VECTOR_STORE_FILE_OBJECT = "vector_store.file"


class VectorStoreFileStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Statuses the worker should not pick up — terminal states.
TERMINAL_FILE_STATUSES: frozenset[str] = frozenset(
    {VectorStoreFileStatus.COMPLETED, VectorStoreFileStatus.CANCELLED}
)


# Allowed prefixes used for OpenAI-style ids.
VECTOR_STORE_ID_PREFIX = "vs_"
VECTOR_STORE_FILE_ID_PREFIX = "file-"
VECTOR_STORE_FILE_BATCH_ID_PREFIX = "vsfb_"
