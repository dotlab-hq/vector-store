from langchain_openai import OpenAIEmbeddings

from src.graph.nodes.retrieval import set_retriever
from src.graph.nodes.reranking import set_reranker
from src.graph.workflows import build_rag_workflow
from src.indexing.bm25.bm25_store import Bm25Store
from src.indexing.embeddings import embeddings
from src.indexing.qdrant.qdrant_store import QdrantVectorStore
from src.observability.logging import get_logger
from src.retrieval.hybrid.hybrid_retriever import HybridRetriever
from src.vector_stores.scheduler import VectorStoreScheduler

logger = get_logger()

# Shared stores — accessible from ingestion pipeline for indexing
qdrant_store: QdrantVectorStore | None = None
bm25_store: Bm25Store | None = None
embedder: OpenAIEmbeddings | None = None
_workflow = None
_scheduler: VectorStoreScheduler | None = None


def init_dependencies() -> None:
    """Initialize all workflow dependencies (retriever, reranker, stores)."""
    global qdrant_store, bm25_store, embedder

    qdrant_store = QdrantVectorStore()
    bm25_store = Bm25Store()
    embedder = embeddings

    hybrid_retriever = HybridRetriever(qdrant_store, bm25_store, embedder)

    set_retriever(hybrid_retriever)

    # Lazy-load reranker — sentence-transformers + torch are heavy optional deps
    try:
        from src.retrieval.reranking.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker()
    except ImportError:
        reranker = None
        logger.info("reranker_disabled", reason="sentence-transformers not installed")
    set_reranker(reranker)


async def check_service_health() -> None:
    """Log availability of all external services. Called once at startup."""
    from src.config import settings

    qdrant_ok = False
    if qdrant_store is not None:
        try:
            qdrant_ok = qdrant_store._client is not None
        except Exception:
            pass

    neo4j_ok = False
    if settings.neo4j_enabled:
        try:
            from src.graph.knowledge.neo4j_client import Neo4jClient

            neo4j_client = Neo4jClient()
            neo4j_ok = await neo4j_client.verify_connectivity()
            await neo4j_client.close()
        except Exception:
            neo4j_ok = False

    logger.info(
        "service_health",
        qdrant=qdrant_ok,
        neo4j=neo4j_ok,
        neo4j_enabled=settings.neo4j_enabled,
        s3=settings.s3_access_key != "",
    )


async def rebuild_bm25() -> None:
    """Rebuild BM25 index from database chunks. Called at startup."""
    global bm25_store
    if bm25_store is None:
        return
    try:
        count = await bm25_store.rebuild_from_db()
        logger.info("bm25_rebuilt", chunk_count=count)
    except Exception as exc:
        logger.warning("bm25_rebuild_failed", error=str(exc))


def init_vector_store_scheduler() -> None:
    """Initialize the background vector store scheduler (worker + cron)."""
    global _scheduler
    _scheduler = VectorStoreScheduler()


def get_scheduler() -> VectorStoreScheduler:
    """Return the vector store scheduler. Must be called after init_vector_store_scheduler."""
    if _scheduler is None:
        raise RuntimeError("VectorStoreScheduler not initialized")
    return _scheduler


def get_workflow():
    """Get the compiled LangGraph RAG workflow, initializing if needed."""
    global _workflow
    if _workflow is None:
        _workflow = build_rag_workflow()
    return _workflow
