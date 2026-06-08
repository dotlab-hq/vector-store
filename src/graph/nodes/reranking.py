from src.config import settings
from src.graph.state.schemas import RAGState, RerankedChunkSnapshot
from src.observability.logging import get_logger

logger = get_logger()

_reranker = None


def set_reranker(reranker) -> None:
    global _reranker
    _reranker = reranker


async def reranking(state: RAGState) -> dict:
    if not state.retrieval_results:
        return {"reranked_results": [], "reranked_chunks": []}

    if _reranker is None:
        # Reranker unavailable — pass retrieval results through unchanged
        logger.info("reranker_skipped", result_count=len(state.retrieval_results))
        return {"reranked_results": list(state.retrieval_results), "reranked_chunks": []}

    query = state.rewritten_query or state.original_query
    reranked = await _reranker.rerank(query, state.retrieval_results, top_k=settings.rerank_top_k)

    # Build reranked-chunk snapshot
    reranked_snapshots = [
        RerankedChunkSnapshot(chunk_id=r.chunk.id, rank=rank, score=float(r.score))
        for rank, r in enumerate(reranked, start=1)
    ]

    logger.info("reranking_done", input_count=len(state.retrieval_results), output_count=len(reranked))
    return {"reranked_results": reranked, "reranked_chunks": reranked_snapshots}
