from src.graph.state.schemas import RAGState
from src.observability.logging import get_logger

logger = get_logger()


def intent_router(state: RAGState) -> str:
    """Route based on query intent. Currently all intents share the same retrieval path.

    Future: kg_query → graph traversal, comparative → side-by-side retrieval, etc.
    """
    logger.info("intent_routed", intent=state.intent.value)
    return "retrieval"
