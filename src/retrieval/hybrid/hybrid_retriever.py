import asyncio
from collections.abc import Sequence

from langchain_openai import OpenAIEmbeddings
from src.config import settings
from src.indexing.bm25.bm25_store import Bm25Store
from src.indexing.qdrant.qdrant_store import QdrantVectorStore
from src.observability.logging import get_logger
from src.retrieval.dense.retriever import DenseRetriever
from src.retrieval.hybrid.fusion import reciprocal_rank_fusion
from src.retrieval.sparse.retriever import SparseRetriever
from src.shared.types import RetrievalResult

logger = get_logger()


class HybridRetriever:
    def __init__(
        self,
        qdrant_store: QdrantVectorStore,
        bm25_store: Bm25Store,
        embedder: OpenAIEmbeddings,
    ) -> None:
        self.dense = DenseRetriever(qdrant_store, embedder)
        self.sparse = SparseRetriever(bm25_store)

    async def retrieve(
        self, query: str, top_k: int = settings.retrieval_top_k
    ) -> Sequence[RetrievalResult]:
        # Run dense and sparse retrieval in parallel
        dense_coro = self.dense.retrieve(query, top_k=top_k)
        sparse_coro = self.sparse.retrieve(query, top_k=top_k)

        dense_results, sparse_results = await asyncio.gather(dense_coro, sparse_coro)

        dense_list = list(dense_results)
        sparse_list = list(sparse_results)

        fused = reciprocal_rank_fusion([dense_list, sparse_list])

        logger.info(
            "hybrid_retrieval",
            query=query[:80],
            dense_count=len(dense_list),
            sparse_count=len(sparse_list),
            fused_count=len(fused),
        )
        return fused[:top_k]
