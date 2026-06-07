from collections import defaultdict

from src.config import settings
from src.shared.types import RetrievalResult


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievalResult]],
    k: int = settings.fusion_k,
) -> list[RetrievalResult]:
    scores: dict[str, float] = defaultdict(float)
    chunks_by_id: dict[str, RetrievalResult] = {}

    for result_list in result_lists:
        for rank, result in enumerate(result_list):
            chunk_id = result.chunk.id
            scores[chunk_id] += 1.0 / (k + rank + 1)
            if chunk_id not in chunks_by_id:
                chunks_by_id[chunk_id] = result

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    fused: list[RetrievalResult] = []
    for chunk_id, score in ranked:
        original = chunks_by_id[chunk_id]
        fused.append(RetrievalResult(chunk=original.chunk, score=score, source="fused"))
    return fused
