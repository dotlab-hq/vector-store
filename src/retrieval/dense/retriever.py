from collections.abc import Sequence

from langchain_openai import OpenAIEmbeddings
from src.indexing.qdrant.qdrant_store import QdrantVectorStore
from src.observability.logging import get_logger
from src.shared.types import RetrievalResult

logger = get_logger()


class DenseRetriever:
    def __init__(self, store: QdrantVectorStore, embedder: OpenAIEmbeddings) -> None:
        self.store = store
        self.embedder = embedder

    async def retrieve(self, query: str, top_k: int = 20) -> Sequence[RetrievalResult]:
        query_embedding = await self.embedder.aembed_query(query)
        results = await self.store.search(query_embedding, top_k=top_k)
        logger.info("dense_retrieval", query=query[:80], results_count=len(results))
        return results
