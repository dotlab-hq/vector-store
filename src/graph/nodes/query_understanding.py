from src.agents.query_agent.agent import query_understanding_node
from src.graph.state.schemas import RAGState
from src.observability.logging import get_logger

logger = get_logger()


async def query_understanding(state: RAGState) -> dict:
    result = await query_understanding_node(state.original_query)

    # Build the query_rewrites list (original + rewritten + sub-queries)
    rewrites: list[str] = [state.original_query]
    rewritten = result["rewritten_query"]
    if rewritten and rewritten != state.original_query:
        rewrites.append(rewritten)
    sub_queries = (
        result["decomposed_query"].sub_queries
        if result.get("decomposed_query")
        else [rewritten]
    )
    for sq in sub_queries:
        if sq not in rewrites:
            rewrites.append(sq)

    logger.info("query_understanding_done", intent=result["intent"].value)
    return {
        "intent": result["intent"],
        "rewritten_query": rewritten,
        "decomposed_query": result.get("decomposed_query"),
        "sub_queries": sub_queries,
        "query_rewrites": rewrites,
    }
