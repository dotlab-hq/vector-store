import asyncio
import json

from src.agents.query_agent.prompts import (
    INTENT_CLASSIFICATION_PROMPT,
    QUERY_DECOMPOSITION_PROMPT,
    QUERY_REWRITING_PROMPT,
)
from src.agents.query_agent.schemas import (
    DecomposedQuery,
    QueryClassification,
    RewrittenQuery,
)
from src.generation.prompts.safe_format import (
    ANTI_INJECTION_SYSTEM_PREAMBLE,
    format_prompt_with_user_data,
)
from src.llm import llm
from src.observability.logging import get_logger
from src.shared.types import QueryIntent

logger = get_logger()


async def _classify_intent(query: str) -> QueryClassification:
    prompt = format_prompt_with_user_data(INTENT_CLASSIFICATION_PROMPT, user_data=query)
    response = await llm.ainvoke([
        ("system", ANTI_INJECTION_SYSTEM_PREAMBLE + " You are a precise classifier."),
        ("human", prompt),
    ])
    try:
        data = json.loads(response.content)
        return QueryClassification(**data)
    except (json.JSONDecodeError, ValueError):
        return QueryClassification(intent=QueryIntent.SIMPLE, confidence=0.5)


async def _rewrite_query(query: str) -> str:
    prompt = format_prompt_with_user_data(QUERY_REWRITING_PROMPT, user_data=query)
    response = await llm.ainvoke([
        ("system", ANTI_INJECTION_SYSTEM_PREAMBLE + " You are a query rewriting engine."),
        ("human", prompt),
    ])
    try:
        data = json.loads(response.content)
        return RewrittenQuery(**data).rewritten
    except (json.JSONDecodeError, ValueError):
        return query


async def _decompose_query(query: str) -> DecomposedQuery:
    prompt = format_prompt_with_user_data(QUERY_DECOMPOSITION_PROMPT, user_data=query)
    response = await llm.ainvoke([
        ("system", ANTI_INJECTION_SYSTEM_PREAMBLE + " You are a query decomposition engine."),
        ("human", prompt),
    ])
    try:
        data = json.loads(response.content)
        return DecomposedQuery(**data)
    except (json.JSONDecodeError, ValueError):
        return DecomposedQuery(original=query, sub_queries=[query])


async def query_understanding_node(query: str) -> dict:
    # Run classification and rewriting in parallel (no dependency between them)
    classification, rewritten = await asyncio.gather(
        _classify_intent(query),
        _rewrite_query(query),
    )

    logger.info(
        "query_classified",
        intent=classification.intent.value,
        confidence=classification.confidence,
    )

    decomposed = None
    if classification.intent in {QueryIntent.MULTI_HOP, QueryIntent.COMPARATIVE, QueryIntent.ANALYTICAL}:
        decomposed = await _decompose_query(rewritten)
        logger.info("query_decomposed", sub_query_count=len(decomposed.sub_queries))

    return {
        "intent": classification.intent,
        "rewritten_query": rewritten,
        "decomposed_query": decomposed,
        "original_query": query,
    }
