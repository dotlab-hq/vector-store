from src.generation.context.builder import ContextBuilder
from src.graph.state.schemas import RAGState
from src.observability.logging import get_logger

logger = get_logger()

_context_builder = ContextBuilder()


async def context_building(state: RAGState) -> dict:
    if not state.reranked_results:
        return {"context": "", "citations": [], "supporting_chunks": []}

    context, supporting = _context_builder.build(list(state.reranked_results))

    logger.info(
        "context_built", chunk_count=len(supporting), context_length=len(context)
    )
    return {
        "context": context,
        "supporting_chunks": supporting,
        # Keep ``citations`` (list[str]) populated for backward compatibility — these
        # are simple ``[chunk_id]`` references. Structured citations are built in
        # the ``generation`` node and finalised in the API route.
        "citations": [f"[{cid}]" for cid in supporting],
    }
