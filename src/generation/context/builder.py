from src.config import settings
from src.shared.types import RetrievalResult


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


class ContextBuilder:
    """Builds a numbered context string the LLM can cite from.

    Each chunk becomes a block like::

        [1] (chunk_id: tmp4god0rsn_p0_c1)
        <chunk content>

    The numbering is the basis for the citation ids the ``CitationBuilder``
    will emit, so an LLM that writes ``[1]`` in its answer maps cleanly to
    ``citations[0]``.
    """

    def __init__(self, max_tokens: int = settings.max_context_tokens) -> None:
        self.max_tokens = max_tokens

    def build(self, results: list[RetrievalResult]) -> tuple[str, list[str]]:
        context_parts: list[str] = []
        supporting: list[str] = []
        token_count = 0

        for idx, result in enumerate(results, start=1):
            content = result.chunk.content
            chunk_tokens = _estimate_tokens(content)

            if token_count + chunk_tokens > self.max_tokens:
                break

            header = f"[{idx}] (chunk_id: {result.chunk.id})"
            if result.chunk.image_url:
                header += f"\n[diagram_image: {result.chunk.image_url}]"
            context_parts.append(f"{header}\n{content}")
            supporting.append(result.chunk.id)
            token_count += chunk_tokens

        return "\n\n---\n\n".join(context_parts), supporting
