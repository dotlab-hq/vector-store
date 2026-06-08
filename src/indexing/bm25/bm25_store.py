from collections.abc import Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from src.shared.types import Chunk, RetrievalResult


class Bm25Store:
    def __init__(self) -> None:
        self.corpus: list[str] = []
        self.chunk_ids: list[str] = []
        self._metadata: dict[str, Chunk] = {}
        self._bm25: BM25Okapi | None = None

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    async def rebuild_from_db(self) -> int:
        """Reload all chunks from the database and rebuild the BM25 index.

        Called at startup so the in-memory index survives server restarts.
        Returns the number of chunks loaded.
        """
        from sqlalchemy import select
        from src.database.session import async_session_factory
        from src.database.models import ChunkModel

        async with async_session_factory() as session:
            result = await session.execute(select(ChunkModel.id, ChunkModel.content))
            rows = result.fetchall()

        self.corpus = [row.content for row in rows]
        self.chunk_ids = [row.id for row in rows]
        self._metadata = {
            row.id: Chunk(id=row.id, document_id="", content=row.content)
            for row in rows
        }
        self._rebuild()
        return len(rows)

    async def search(self, query: str, top_k: int = 10) -> Sequence[RetrievalResult]:
        if self._bm25 is None or not self.corpus:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[RetrievalResult] = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            chunk_id = self.chunk_ids[idx]
            chunk = self._metadata.get(chunk_id)
            if chunk:
                results.append(
                    RetrievalResult(
                        chunk=chunk, score=float(scores[idx]), source="bm25"
                    )
                )
        return results

    async def insert(self, chunks: Sequence[Chunk]) -> None:
        for chunk in chunks:
            self.corpus.append(chunk.content)
            self.chunk_ids.append(chunk.id)
            self._metadata[chunk.id] = chunk
        self._rebuild()

    async def delete(self, chunk_ids: Sequence[str]) -> None:
        ids_to_delete = set(chunk_ids)
        new_corpus = []
        new_ids = []
        new_metadata: dict[str, Chunk] = {}

        for cid, content in zip(self.chunk_ids, self.corpus):
            if cid not in ids_to_delete:
                new_corpus.append(content)
                new_ids.append(cid)
                if cid in self._metadata:
                    new_metadata[cid] = self._metadata[cid]

        self.corpus = new_corpus
        self.chunk_ids = new_ids
        self._metadata = new_metadata
        self._rebuild()

    async def count(self) -> int:
        return len(self.corpus)

    def _rebuild(self) -> None:
        if self.corpus:
            tokenized = [self._tokenize(doc) for doc in self.corpus]
            self._bm25 = BM25Okapi(tokenized)
        else:
            self._bm25 = None
