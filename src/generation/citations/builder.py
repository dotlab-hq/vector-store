from pydantic import BaseModel

from src.shared.types import RetrievalResult


class CitationItem(BaseModel):
    """A single citation pointing at a specific chunk."""

    id: int
    source_id: str
    chunk_id: str
    document_id: str
    document_name: str
    page: int | None = None
    title: str = ""
    snippet: str = ""
    relevance_score: float = 0.0


class SourceItem(BaseModel):
    """A deduplicated document-level source referenced by one or more citations."""

    source_id: str
    document_id: str
    document_name: str
    page: int | None = None
    title: str = ""
    url: str | None = None


def _snippet(content: str, max_chars: int = 220) -> str:
    text = " ".join(content.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _source_id(chunk_id: str) -> str:
    """Derive a page/document-level source id from a chunk id.

    Examples:
        tmp4god0rsn_p0_c1       -> tmp4god0rsn_p0
        tmp4god0rsn_p12_c3      -> tmp4god0rsn_p12
        tmp4god0rsn_p0          -> tmp4god0rsn_p0
    """
    # Strip the trailing ``_c<digits>`` child-chunk marker if present.
    if "_c" in chunk_id:
        head, tail = chunk_id.rsplit("_c", 1)
        if tail.isdigit():
            return head
    return chunk_id


class CitationBuilder:
    """Builds structured citations and deduped source lists from retrieval results.

    Citations are numbered in the order results are received, which means the
    citation ids match the numbering used in the prompt context — so a model
    answer that references ``[1]`` naturally maps to ``citations[0]``.
    """

    def __init__(self, document_titles: dict[str, str] | None = None) -> None:
        self.document_titles = document_titles or {}

    def build(
        self,
        results: list[RetrievalResult],
    ) -> tuple[list[CitationItem], list[SourceItem]]:
        citations: list[CitationItem] = []
        sources_by_id: dict[str, SourceItem] = {}

        for idx, result in enumerate(results, start=1):
            chunk = result.chunk
            document_id = chunk.document_id
            document_name = self.document_titles.get(document_id, document_id)
            source_id = _source_id(chunk.id)

            citation = CitationItem(
                id=idx,
                source_id=source_id,
                chunk_id=chunk.id,
                document_id=document_id,
                document_name=document_name,
                page=chunk.page_number,
                title=document_name,
                snippet=_snippet(chunk.content),
                relevance_score=float(result.score),
            )
            citations.append(citation)

            if source_id not in sources_by_id:
                sources_by_id[source_id] = SourceItem(
                    source_id=source_id,
                    document_id=document_id,
                    document_name=document_name,
                    page=chunk.page_number,
                    title=document_name,
                    url=None,
                )

        return citations, list(sources_by_id.values())
