from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.graph.nodes.context_building import context_building
from src.graph.nodes.faithfulness import faithfulness_check
from src.graph.nodes.generation import generation
from src.graph.nodes.query_understanding import query_understanding
from src.graph.nodes.reranking import reranking
from src.graph.nodes.retrieval import retrieval
from src.graph.state.schemas import RAGState


def faithfulness_router(state: RAGState) -> str:
    if state.faithfulness_passed or state.claim_count == 0:
        return "end"
    return "regenerate"


async def regeneration(state: RAGState) -> dict:
    """Regenerate when faithfulness check fails."""
    from src.generation.prompts.safe_format import (
        ANTI_INJECTION_SYSTEM_PREAMBLE,
        fence_user_data,
    )
    from src.llm import llm

    system_prompt = (
        ANTI_INJECTION_SYSTEM_PREAMBLE
        + " You are a precise, citation-backed assistant. Answer the user's question using ONLY the provided context."
        ' If the context does not contain enough information, say "I don\'t have sufficient evidence to answer this question."'
        " Never use external knowledge. Cite sources using [1], [2] inline markers matching the 1-based position in the context block."
        " Be concise and factual. If you cannot confidently answer from the context, do not answer."
        " Output JSON with keys: answer, short_answer."
    )

    context_block = fence_user_data(state.context or "")
    question_block = fence_user_data(
        state.rewritten_query or state.original_query or ""
    )

    human_prompt = (
        f"Context:\n{context_block}\n\n"
        f"Question:\n{question_block}\n\n"
        "The previous answer was not faithful to the context. Please provide a more accurate answer grounded in the provided context only:"
    )

    response = await llm.ainvoke(
        [
            ("system", system_prompt),
            ("human", human_prompt),
        ]
    )

    import json as _json
    import re as _re

    raw = response.content or ""
    cleaned = _re.sub(r"```(?:json)?\s*", "", raw)
    cleaned = _re.sub(r"```\s*$", "", cleaned.strip())
    try:
        data = _json.loads(cleaned) if cleaned else {}
    except _json.JSONDecodeError:
        data = {}

    answer = str(data.get("answer") or "").strip() or raw.strip()
    short_answer = str(data.get("short_answer") or "").strip()

    return {"response": answer, "short_answer": short_answer}


def build_rag_workflow() -> CompiledStateGraph:
    graph = StateGraph(RAGState)

    # Add nodes
    graph.add_node("query_understanding", query_understanding)
    graph.add_node("retrieval", retrieval)
    graph.add_node("reranking", reranking)
    graph.add_node("context_building", context_building)
    graph.add_node("generation", generation)
    graph.add_node("faithfulness_check", faithfulness_check)
    graph.add_node("regeneration", regeneration)

    # Edges
    graph.add_edge(START, "query_understanding")
    graph.add_edge("query_understanding", "retrieval")
    graph.add_edge("retrieval", "reranking")
    graph.add_edge("reranking", "context_building")
    graph.add_edge("context_building", "generation")
    graph.add_edge("generation", "faithfulness_check")

    graph.add_conditional_edges(
        "faithfulness_check",
        faithfulness_router,
        {
            "end": END,
            "regenerate": "regeneration",
        },
    )

    graph.add_edge("regeneration", END)

    return graph.compile()
