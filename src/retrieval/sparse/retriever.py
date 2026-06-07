from collections.abc import Sequence

from src.indexing.bm25.bm25_store import Bm25Store
from src.observability.logging import get_logger
from src.shared.types import RetrievalResult

logger = get_logger()


class SparseRetriever:
    def __init__(self, store: Bm25Store) -> None:
        self.store = store

    async def retrieve(self, query: str, top_k: int = 20) -> Sequence[RetrievalResult]:
        results = await self.store.search(query, top_k=top_k)
        logger.info("sparse_retrieval", query=query[:80], results_count=len(results))
        return results
