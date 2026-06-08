"""Reranker using Hugging Face Inference API — no local model download needed."""

from collections.abc import Sequence

import httpx

from src.config import settings
from src.observability.logging import get_logger
from src.shared.types import RetrievalResult

logger = get_logger()

HF_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
HF_INFERENCE_URL = f"https://api-inference.huggingface.co/pipeline/rerank/{HF_RERANK_MODEL}"


class HfReranker:
    """Reranker backed by HF Inference API (free tier works with a token)."""

    def __init__(self, api_token: str | None = None) -> None:
        self._token = api_token or settings.hf_token or ""
        if not self._token:
            logger.warning("hf_reranker_no_token", detail="HF_TOKEN not set — unauthenticated requests have low rate limits")

    async def rerank(
        self,
        query: str,
        results: Sequence[RetrievalResult],
        top_k: int = settings.rerank_top_k,
    ) -> list[RetrievalResult]:
        if not results:
            return []

        inputs = [r.chunk.content for r in results]

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                HF_INFERENCE_URL,
                headers={"Authorization": f"Bearer {self._token}"} if self._token else {},
                json={"query": query, "texts": inputs, "raw_scores": False},
            )
            resp.raise_for_status()
            data = resp.json()

        # HF returns list of {index, score} sorted descending
        reranked: list[RetrievalResult] = []
        for item in data:
            idx = item["index"]
            score = float(item["score"])
            reranked.append(
                RetrievalResult(
                    chunk=results[idx].chunk,
                    score=score,
                    source="reranked",
                )
            )

        top = reranked[:top_k]
        logger.info(
            "remote_reranking",
            model=HF_RERANK_MODEL,
            input_count=len(results),
            output_count=len(top),
        )
        return top
