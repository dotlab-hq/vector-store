"""Factory for creating the appropriate retriever based on configuration.

Supports:
- ``local``: HybridRetriever (Qdrant dense + BM25 sparse + RRF fusion)
- ``kg``: KGRetriever (Neo4j knowledge graph traversal)
"""

from __future__ import annotations

from src.config import settings
from src.observability.logging import get_logger

logger = get_logger()


def create_retriever() -> object:
    """Create and return a retriever matching ``settings.retriever_type``.

    Returns a concrete retriever instance whose ``retrieve()`` method
    satisfies the :class:`Retriever` protocol.
    """
    retriever_type = settings.retriever_type

    if retriever_type == "local":
        from src.indexing.bm25.bm25_store import Bm25Store
        from src.indexing.embeddings import embeddings
        from src.indexing.qdrant.qdrant_store import QdrantVectorStore
        from src.retrieval.hybrid.hybrid_retriever import HybridRetriever

        qdrant_store = QdrantVectorStore()
        bm25_store = Bm25Store()
        hybrid = HybridRetriever(qdrant_store, bm25_store, embeddings)
        logger.info("retriever_created", type="local_hybrid")
        return hybrid

    if retriever_type == "kg":
        from src.graph.knowledge.graph_store import KnowledgeGraphStore
        from src.graph.knowledge.neo4j_client import Neo4jClient
        from src.retrieval.kg.retriever import KGRetriever

        neo4j = Neo4jClient()
        graph_store = KnowledgeGraphStore(neo4j)
        kg = KGRetriever(graph_store)
        logger.info("retriever_created", type="knowledge_graph")
        return kg

    raise ValueError(
        f"Unknown retriever_type {retriever_type!r}. "
        f"Supported: 'local', 'kg'."
    )
