import json
import re

from src.graph.state.schemas import RAGState
from src.llm import llm
from src.generation.prompts.safe_format import (
    fence_user_data,
    ANTI_INJECTION_SYSTEM_PREAMBLE,
)
from src.observability.logging import get_logger

logger = get_logger()

CLAIM_EXTRACTION_PROMPT = """Extract all factual claims from this answer. Each claim should be a single verifiable statement.
Return a JSON array of strings, each being one claim.

Answer: {answer}"""

VERIFICATION_PROMPT = """You are a faithfulness verifier. For each claim, determine if it is supported by the given context.
Return a JSON array of objects with: claim (string), supported (boolean), reasoning (string).

Context: {context}

Claims: {claims}"""


def _parse_llm_json(raw: str) -> list:
    """Parse LLM JSON output, handling markdown code fences."""
    # Strip ```json ... ``` wrapping if present
    cleaned = re.sub(r"```(?:json)?\s*", "", raw)
    cleaned = re.sub(r"```\s*$", "", cleaned.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError, ValueError:
        return []


def _extract_citation_markers(answer: str) -> set[int]:
    """Find all ``[N]`` markers in the answer text (1-based)."""
    return {int(m) for m in re.findall(r"\[(\d+)\]", answer or "")}


def _estimate_context_recall(answer: str, supporting_chunks: list[str]) -> float:
    """Context recall ≈ fraction of supporting chunks actually cited.

    If the model cited at least one marker, this is the fraction of supporting
    chunks referenced; if it cited nothing, recall is 0.
    """
    if not supporting_chunks:
        return 0.0
    markers = _extract_citation_markers(answer)
    if not markers:
        return 0.0
    return min(1.0, len(markers) / max(1, len(supporting_chunks)))


def _estimate_answer_relevance(answer: str, query: str) -> float:
    """Cheap lexical relevance score: token overlap between answer and query.

    Returns 0.0-1.0. Only tokens length >= 3 are considered (rough stopword filter).
    """
    if not answer or not query:
        return 0.0
    q_tokens = {t for t in re.findall(r"\w+", query.lower()) if len(t) >= 3}
    a_tokens = set(re.findall(r"\w+", answer.lower()))
    if not q_tokens:
        return 0.0
    overlap = len(q_tokens & a_tokens)
    return min(1.0, overlap / len(q_tokens))


async def faithfulness_check(state: RAGState) -> dict:
    if not state.context or not state.response:
        return {
            "faithfulness_score": 0.0,
            "faithfulness_passed": False,
            "claim_count": 0,
            "supported_claims": 0,
            "answer_relevance_score": 0.0,
            "context_recall_score": 0.0,
            "confidence": 0.0,
        }

    # Step 1: Extract claims
    fenced_answer = fence_user_data(state.response)
    extract_prompt = CLAIM_EXTRACTION_PROMPT.format(answer=fenced_answer)
    extract_response = await llm.ainvoke(
        [
            ("system", f"{ANTI_INJECTION_SYSTEM_PREAMBLE} Extract claims precisely."),
            ("human", extract_prompt),
        ]
    )

    claims = _parse_llm_json(extract_response.content)

    if not claims:
        # Fall back to a softer scoring path so we still emit evaluation numbers
        rel = _estimate_answer_relevance(state.response, state.original_query)
        recall = _estimate_context_recall(state.response, state.supporting_chunks)
        confidence = round((rel + recall) / 2, 4)
        return {
            "faithfulness_score": 0.5,
            "faithfulness_passed": False,
            "claim_count": 0,
            "supported_claims": 0,
            "answer_relevance_score": round(rel, 4),
            "context_recall_score": round(recall, 4),
            "confidence": confidence,
        }

    # Step 2: Verify each claim
    fenced_context = fence_user_data(state.context)
    fenced_claims = fence_user_data(json.dumps(claims))
    verify_prompt = VERIFICATION_PROMPT.format(
        context=fenced_context,
        claims=fenced_claims,
    )
    verify_response = await llm.ainvoke(
        [
            (
                "system",
                f"{ANTI_INJECTION_SYSTEM_PREAMBLE} Verify claims against context.",
            ),
            ("human", verify_prompt),
        ]
    )

    verifications = _parse_llm_json(verify_response.content)

    supported = sum(1 for v in verifications if v.get("supported", False))
    total = len(claims)
    score = supported / total if total > 0 else 0.5
    passed = score >= 0.8

    # Estimate the additional metrics
    answer_relevance = _estimate_answer_relevance(state.response, state.original_query)
    context_recall = _estimate_context_recall(state.response, state.supporting_chunks)

    # Confidence is a weighted blend: faithfulness is the strongest signal, with
    # relevance and recall nudging the final number for the user.
    confidence = round(
        0.6 * score + 0.25 * answer_relevance + 0.15 * context_recall,
        4,
    )

    logger.info(
        "faithfulness_check_done",
        total_claims=total,
        supported_claims=supported,
        faithfulness=score,
        answer_relevance=answer_relevance,
        context_recall=context_recall,
        confidence=confidence,
        passed=passed,
    )

    return {
        "faithfulness_score": score,
        "faithfulness_passed": passed,
        "claim_count": total,
        "supported_claims": supported,
        "answer_relevance_score": round(answer_relevance, 4),
        "context_recall_score": round(context_recall, 4),
        "confidence": confidence,
    }
