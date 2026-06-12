"""Qdrant-backed vector store implementation.

Replaces the former FAISS in-process store with a production-ready
Qdrant deployment that supports dense, sparse, and hybrid retrieval.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import cast
from uuid import uuid4

import numpy as np
from qdrant_client import QdrantClient, models as qm
from qdrant_client.http.models import Distance, VectorParams

from src.config import settings
from src.observability.logging import get_logger
from src.shared.base import VectorStore
from src.shared.types import Chunk, RetrievalResult

logger = get_logger()

COLLECTION_NAME = "rag_chunks"
VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


class QdrantVectorStore(VectorStore):
    """Production-ready vector store backed by Qdrant.

    Features:
    - Dense cosine similarity search via ``QdrantVectorStore``.
    - Store-scoped and predicate-filtered search (``search_in_stores``).
    - Automatic collection creation on first use.
    """

    def __init__(self, dimension: int = settings.embedding_dimension) -> None:
        self.dimension = dimension
        self._uses_named_vectors = True

        # Prefer a running Qdrant server; fall back to in-memory for dev
        if settings.qdrant_url:
            self._client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key or None,
                prefer_grpc=settings.qdrant_prefer_grpc,
            )
            logger.info(
                "qdrant_connected",
                url=settings.qdrant_url,
                collection=COLLECTION_NAME,
            )
        elif settings.qdrant_path:
            self._client = QdrantClient(path=settings.qdrant_path)
            logger.info(
                "qdrant_local",
                path=settings.qdrant_path,
                collection=COLLECTION_NAME,
            )
        else:
            self._client = QdrantClient(":memory:")
            logger.info("qdrant_in_memory", collection=COLLECTION_NAME)

        self._ensure_collection()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""
        collections = [c.name for c in self._client.get_collections().collections]
        if COLLECTION_NAME not in collections:
            self._client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={
                    VECTOR_NAME: VectorParams(
                        size=self.dimension,
                        distance=Distance.COSINE,
                    )
                },
            )
            self._uses_named_vectors = True
            logger.info(
                "qdrant_collection_created",
                collection=COLLECTION_NAME,
                dimension=self.dimension,
                vector_name=VECTOR_NAME,
            )
        else:
            self._uses_named_vectors = self._collection_uses_named_dense_vector()

        # Ensure indexes on payload fields for filtered search
        self._ensure_index("vector_store_id")
        self._ensure_index("document_id")

    def _collection_uses_named_dense_vector(self) -> bool:
        """Return whether the collection exposes the configured dense vector name."""
        info = self._client.get_collection(COLLECTION_NAME)
        vectors_config = info.config.params.vectors
        if isinstance(vectors_config, dict):
            if VECTOR_NAME not in vectors_config:
                available = sorted(str(name) for name in vectors_config)
                raise ValueError(
                    f"Qdrant collection {COLLECTION_NAME!r} does not contain "
                    f"required vector name {VECTOR_NAME!r}. Available vectors: "
                    f"{available}"
                )
            return True

        # Existing development collections were created with a single unnamed
        # vector. Keep them readable/writable instead of failing searches with
        # "Not existing vector name error: dense".
        logger.warning(
            "qdrant_unnamed_vector_collection",
            collection=COLLECTION_NAME,
            expected_vector_name=VECTOR_NAME,
        )
        return False

    def _query_kwargs(self, query_embedding: list[float]) -> dict:
        """Build Qdrant query kwargs for named or legacy unnamed collections."""
        kwargs: dict = {
            "collection_name": COLLECTION_NAME,
            "query": query_embedding,
        }
        if self._uses_named_vectors:
            kwargs["using"] = VECTOR_NAME
        return kwargs

    def _point_vector(self, vector: list[float]) -> qm.VectorStruct:
        """Build a Qdrant point vector for named or legacy unnamed collections."""
        if self._uses_named_vectors:
            return cast(qm.VectorStruct, {VECTOR_NAME: vector})
        return cast(qm.VectorStruct, vector)

    async def _run_qdrant(self, fn, *args, **kwargs):  # noqa: ANN002
        """Run a synchronous Qdrant client method in a thread to avoid blocking."""
        return await asyncio.to_thread(fn, *args, **kwargs)

    def _ensure_index(self, field_name: str) -> None:
        """Create an index on a field if it doesn't exist."""
        try:
            self._client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_type=qm.PayloadSchemaType.KEYWORD,
            )
            logger.info("qdrant_index_created", field=field_name)
        except Exception as e:
            # Index may already exist; log but don't fail
            logger.debug("qdrant_index_check", field=field_name, error=str(e))

    @staticmethod
    def _chunk_payload(chunk: Chunk) -> dict:
        """Serialize a ``Chunk`` into a Qdrant payload dict."""
        return {
            "id": chunk.id,
            "document_id": chunk.document_id,
            "content": chunk.content,
            "parent_id": chunk.parent_id,
            "page_number": chunk.page_number,
            "position": chunk.position,
            "section": chunk.section,
            "vector_store_id": chunk.vector_store_id or "",
            "attributes": chunk.attributes,
            "image_url": chunk.image_url or "",
        }

    @staticmethod
    def _payload_to_chunk(payload: dict) -> Chunk:
        """Deserialize a Qdrant payload dict back into a ``Chunk``."""
        return Chunk(
            id=payload.get("id", ""),
            document_id=payload.get("document_id", ""),
            content=payload.get("content", ""),
            parent_id=payload.get("parent_id"),
            page_number=payload.get("page_number"),
            position=payload.get("position", 0),
            section=payload.get("section", ""),
            vector_store_id=payload.get("vector_store_id") or None,
            attributes=payload.get("attributes", {}),
            image_url=payload.get("image_url") or None,
        )

    # ------------------------------------------------------------------
    # VectorStore interface
    # ------------------------------------------------------------------

    async def search(
        self, query_embedding: list[float], top_k: int = 10
    ) -> Sequence[RetrievalResult]:
        if not query_embedding:
            return []

        results = await self._run_qdrant(
            self._client.query_points,
            **self._query_kwargs(query_embedding),
            limit=top_k,
            with_payload=True,
        )

        out: list[RetrievalResult] = []
        for point in results.points:
            payload = point.payload or {}
            chunk = self._payload_to_chunk(payload)
            # Qdrant returns cosine distance scores in [0, 1]; higher = better
            out.append(
                RetrievalResult(chunk=chunk, score=float(point.score), source="dense")
            )
        return out

    async def insert(
        self, chunks: Sequence[Chunk], embeddings: Sequence[list[float]]
    ) -> None:
        if not chunks or not embeddings:
            return

        vectors = np.array(embeddings, dtype=np.float32)
        if vectors.ndim != 2:
            logger.error("invalid_embedding_shape", shape=vectors.shape)
            return

        actual_dim = vectors.shape[1]
        if actual_dim != self.dimension:
            logger.warning(
                "dimension_mismatch",
                expected=self.dimension,
                actual=actual_dim,
            )
            self.dimension = actual_dim
            self._ensure_collection()

        # Normalize for cosine similarity (Qdrant handles this internally
        # with Distance.COSINE, but we normalize to be safe / match FAISS
        # behavior from before.)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vectors = vectors / norms

        points = []
        for chunk, vec in zip(chunks, vectors):
            points.append(
                qm.PointStruct(
                    id=str(uuid4()),
                    vector=self._point_vector(vec.tolist()),
                    payload=self._chunk_payload(chunk),
                )
            )

        await self._run_qdrant(
            self._client.upsert,
            collection_name=COLLECTION_NAME,
            points=points,
        )
        logger.debug(
            "qdrant_insert",
            count=len(points),
            dimension=self.dimension,
        )

    async def delete(self, chunk_ids: Sequence[str]) -> None:
        if not chunk_ids:
            return

        await self._run_qdrant(
            self._client.delete,
            collection_name=COLLECTION_NAME,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="id",
                            match=qm.MatchAny(any=list(chunk_ids)),
                        )
                    ]
                )
            ),
        )
        logger.debug("qdrant_delete", count=len(chunk_ids))

    async def delete_by_document_id(self, document_id: str) -> None:
        """Delete all Qdrant points for a document. Used before re-indexing."""
        await self._run_qdrant(
            self._client.delete,
            collection_name=COLLECTION_NAME,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="document_id",
                            match=qm.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )

    async def search_in_stores(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        allowed_store_ids: set[str] | None = None,
        predicate: object | None = None,
    ) -> Sequence[RetrievalResult]:
        """Search scoped to specific vector stores, with optional attribute predicate."""
        if not query_embedding:
            return []

        fanout = settings.vector_store_search_fanout
        fetch_k = top_k * fanout

        # Build Qdrant filter conditions
        must_conditions: list[qm.Condition] = []

        if allowed_store_ids is not None and allowed_store_ids:
            must_conditions.append(
                qm.FieldCondition(
                    key="vector_store_id",
                    match=qm.MatchAny(any=list(allowed_store_ids)),
                )
            )

        qdrant_filter = qm.Filter(must=must_conditions) if must_conditions else None

        results = await self._run_qdrant(
            self._client.query_points,
            **self._query_kwargs(query_embedding),
            limit=fetch_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        out: list[RetrievalResult] = []
        for point in results.points:
            payload = point.payload or {}
            chunk = self._payload_to_chunk(payload)

            # Apply attribute predicate (post-filter)
            if predicate is not None and not callable(predicate):
                # predicate is not usable, skip
                pass
            elif predicate is not None and callable(predicate):
                attrs = chunk.attributes
                if not predicate(attrs):
                    continue

            out.append(
                RetrievalResult(chunk=chunk, score=float(point.score), source="dense")
            )
            if len(out) >= top_k:
                break

        return out

    async def update_chunks_vector_store_id(
        self, document_id: str, vector_store_id: str
    ) -> int:
        """Update the ``vector_store_id`` payload on all Qdrant points belonging to *document_id*.

        Returns the number of points updated.
        """
        # Fetch all matching points (Qdrant has no bulk payload-update, so we
        # scroll, patch, and upsert).
        points, _ = await self._run_qdrant(
            self._client.scroll,
            collection_name=COLLECTION_NAME,
            scroll_filter=qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="document_id",
                        match=qm.MatchValue(value=document_id),
                    )
                ]
            ),
            with_payload=True,
            with_vectors=True,
            limit=10_000,
        )
        if not points:
            return 0

        updated: list[qm.PointStruct] = []
        for p in points:
            payload = dict(p.payload or {})
            if payload.get("vector_store_id") == vector_store_id:
                continue
            payload["vector_store_id"] = vector_store_id
            updated.append(
                qm.PointStruct(
                    id=p.id,
                    vector=p.vector,
                    payload=payload,
                )
            )
        if updated:
            await self._run_qdrant(
                self._client.upsert, collection_name=COLLECTION_NAME, points=updated
            )
        return len(updated)

    async def count(self) -> int:
        info = await self._run_qdrant(self._client.get_collection, COLLECTION_NAME)
        return info.points_count or 0
