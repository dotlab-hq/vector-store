import asyncio

from src.graph.state.schemas import RAGState, RetrievedChunkSnapshot
from src.observability.logging import get_logger

logger = get_logger()

# Global stores — set during workflow compilation
_hybrid_retriever = None


def set_retriever(retriever) -> None:
    global _hybrid_retriever
    _hybrid_retriever = retriever


async def retrieval(state: RAGState) -> dict:
    query = state.rewritten_query or state.original_query
    all_results = []

    # Run all sub-query retrievals in parallel
    async def _retrieve_subquery(sub_q: str):
        return await _hybrid_retriever.retrieve(sub_q)

    tasks = [_retrieve_subquery(sub_q) for sub_q in state.sub_queries]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            logger.error("subquery_retrieval_error", error=str(result))
            continue
        all_results.extend(result)

    # Deduplicate by chunk ID, keep highest score
    best_by_id: dict[str, object] = {}
    for r in all_results:
        existing = best_by_id.get(r.chunk.id)
        if existing is None or r.score > existing.score:
            best_by_id[r.chunk.id] = r

    deduped = list(best_by_id.values())

    # Build retrieved-chunk snapshot in rank order
    ranked = sorted(deduped, key=lambda r: r.score, reverse=True)
    snapshots: list[RetrievedChunkSnapshot] = []
    for rank, r in enumerate(ranked, start=1):
        snapshots.append(
            RetrievedChunkSnapshot(
                chunk_id=r.chunk.id,
                document_id=r.chunk.document_id,
                page=r.chunk.page_number,
                rank=rank,
                score=float(r.score),
                content=r.chunk.content,
            )
        )

    logger.info(
        "retrieval_done",
        query_count=len(state.sub_queries),
        results_count=len(deduped),
    )
    return {"retrieval_results": deduped, "retrieved_chunks": snapshots}
