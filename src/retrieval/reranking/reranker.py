import asyncio
from collections.abc import Sequence

from src.config import settings
from src.observability.logging import get_logger
from src.shared.types import RetrievalResult

logger = get_logger()


class CrossEncoderReranker:
    def __init__(
        self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ) -> None:
        self.model_name = model_name
        self._model = None

    def _load_model(self) -> None:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)

    def _predict_sync(self, pairs) -> list:
        """Synchronous prediction to be run in executor."""
        self._load_model()
        return self._model.predict(pairs)

    async def rerank(
        self,
        query: str,
        results: Sequence[RetrievalResult],
        top_k: int = settings.rerank_top_k,
    ) -> list[RetrievalResult]:
        if not results:
            return []

        pairs = [(query, r.chunk.content) for r in results]

        # Run synchronous model inference in a thread pool
        scores = await asyncio.get_event_loop().run_in_executor(
            None, self._predict_sync, pairs
        )

        reranked: list[RetrievalResult] = []
        for result, score in zip(results, scores):
            reranked.append(
                RetrievalResult(
                    chunk=result.chunk,
                    score=float(score),
                    source="reranked",
                )
            )

        reranked.sort(key=lambda x: x.score, reverse=True)
        top_results = reranked[:top_k]

        logger.info(
            "reranking",
            input_count=len(results),
            output_count=len(top_results),
        )
        return top_results
