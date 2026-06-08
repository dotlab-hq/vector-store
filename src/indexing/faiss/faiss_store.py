import json
from collections.abc import Sequence
from pathlib import Path

import faiss
import numpy as np

from src.config import settings
from src.observability.logging import get_logger
from src.shared.base import VectorStore
from src.shared.types import Chunk, RetrievalResult

logger = get_logger()


class FaissVectorStore(VectorStore):
    def __init__(self, dimension: int = settings.embedding_dimension) -> None:
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunk_ids: list[str] = []
        self._metadata: dict[str, Chunk] = {}
        self._embeddings: list[list[float]] = []  # stored for delete+rebuild
        self._store_to_positions: dict[
            str, set[int]
        ] = {}  # vector_store_id → FAISS positions

    def _ensure_dimension(self, vectors: np.ndarray) -> None:
        """Rebuild the index if the vector dimension doesn't match."""
        if vectors.shape[1] != self.dimension:
            logger.warning(
                "dimension_mismatch",
                expected=self.dimension,
                actual=vectors.shape[1],
                index_ntotal=self.index.ntotal,
                stored_embeddings=len(self._embeddings),
            )
            self.dimension = vectors.shape[1]
            self.index = faiss.IndexFlatIP(self.dimension)
            # Old embeddings have a different dimension and cannot be re-added
            if self._embeddings:
                logger.warning(
                    "clearing_incompatible_embeddings",
                    old_dimension=self.dimension,
                    cleared_count=len(self._embeddings),
                )
                self._embeddings.clear()
                self.chunk_ids.clear()
                self._metadata.clear()
                self._store_to_positions.clear()

    async def search(
        self, query_embedding: list[float], top_k: int = 10
    ) -> Sequence[RetrievalResult]:
        if self.index.ntotal == 0:
            return []

        # Ensure query dimension matches index
        if len(query_embedding) != self.dimension:
            logger.warning(
                "search_dimension_mismatch",
                index_dimension=self.dimension,
                query_dimension=len(query_embedding),
            )
            return []

        query = np.array([query_embedding], dtype=np.float32)
        scores, indices = self.index.search(query, min(top_k, self.index.ntotal))

        results: list[RetrievalResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunk_ids):
                continue
            chunk_id = self.chunk_ids[idx]
            chunk = self._metadata.get(chunk_id)
            if chunk:
                results.append(
                    RetrievalResult(chunk=chunk, score=float(score), source="dense")
                )
        return results

    async def insert(
        self, chunks: Sequence[Chunk], embeddings: Sequence[list[float]]
    ) -> None:
        if not chunks or not embeddings:
            return

        vectors = np.array(embeddings, dtype=np.float32)

        # Validate and adapt dimension if needed
        if vectors.ndim != 2:
            logger.error("invalid_embedding_shape", shape=vectors.shape)
            return

        self._ensure_dimension(vectors)

        faiss.normalize_L2(vectors)

        base_pos = self.index.ntotal

        try:
            self.index.add(vectors)
        except AssertionError:
            logger.warning(
                "add_failed_assertion_rebuilding",
                vector_dim=vectors.shape[1],
                index_dim=self.index.d,
                index_ntotal=self.index.ntotal,
            )
            # Rebuild the index with the actual vector dimension and retry
            self.dimension = vectors.shape[1]
            self.index = faiss.IndexFlatIP(self.dimension)
            if self._embeddings:
                self._embeddings.clear()
                self.chunk_ids.clear()
                self._metadata.clear()
                self._store_to_positions.clear()
            base_pos = 0
            faiss.normalize_L2(vectors)
            self.index.add(vectors)
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            pos = base_pos + i
            self.chunk_ids.append(chunk.id)
            self._metadata[chunk.id] = chunk
            self._embeddings.append(emb)
            # Maintain per-vector-store position map
            store_id = chunk.vector_store_id or ""
            self._store_to_positions.setdefault(store_id, set()).add(pos)

    async def delete(self, chunk_ids: Sequence[str]) -> None:
        ids_to_delete = set(chunk_ids)
        remaining_chunks: list[Chunk] = []
        remaining_embeddings: list[list[float]] = []

        for cid, emb in zip(self.chunk_ids, self._embeddings):
            if cid not in ids_to_delete:
                chunk = self._metadata.get(cid)
                if chunk:
                    remaining_chunks.append(chunk)
                    remaining_embeddings.append(emb)

        # Rebuild index from scratch with remaining data
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunk_ids = []
        self._metadata = {}
        self._embeddings = []
        self._store_to_positions = {}

        if remaining_chunks:
            await self.insert(remaining_chunks, remaining_embeddings)

    async def search_in_stores(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        allowed_store_ids: set[str] | None = None,
        predicate: object | None = None,
    ) -> Sequence[RetrievalResult]:
        """Search FAISS scoped to specific vector stores, with optional attribute predicate.

        Uses a fan-out multiplier (``top_k * fanout``) to compensate for post-filtering.
        """
        if self.index.ntotal == 0:
            return []

        if len(query_embedding) != self.dimension:
            logger.warning(
                "search_in_stores_dimension_mismatch",
                index_dimension=self.dimension,
                query_dimension=len(query_embedding),
            )
            return []

        fanout = settings.vector_store_search_fanout
        fetch_k = min(top_k * fanout, self.index.ntotal)

        query = np.array([query_embedding], dtype=np.float32)
        scores, indices = self.index.search(query, fetch_k)

        results: list[RetrievalResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunk_ids):
                continue
            chunk_id = self.chunk_ids[idx]
            chunk = self._metadata.get(chunk_id)
            if chunk is None:
                continue
            # Filter by allowed vector store IDs
            if allowed_store_ids is not None:
                if (chunk.vector_store_id or "") not in allowed_store_ids:
                    continue
            # Apply attribute predicate
            if predicate is not None and not predicate(chunk.attributes):
                continue
            results.append(
                RetrievalResult(chunk=chunk, score=float(score), source="dense")
            )
            if len(results) >= top_k:
                break
        return results

    async def count(self) -> int:
        return self.index.ntotal

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        # Save metadata alongside the index
        meta = {
            "chunk_ids": self.chunk_ids,
            "dimension": self.dimension,
        }
        meta_path = path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

    def load(self, path: Path) -> None:
        if path.exists():
            self.index = faiss.read_index(str(path))
            self.dimension = self.index.d
        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.chunk_ids = meta.get("chunk_ids", [])
            # If metadata has dimension, prefer it (matches stored embeddings)
            saved_dim = meta.get("dimension")
            if saved_dim is not None:
                self.dimension = saved_dim
