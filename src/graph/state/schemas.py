from pydantic import BaseModel, Field

from src.agents.query_agent.schemas import DecomposedQuery
from src.shared.types import QueryIntent, RetrievalResult


class RetrievedChunkSnapshot(BaseModel):
    """Snapshot of a single retrieved chunk used to build the response payload."""

    chunk_id: str
    document_id: str
    page: int | None = None
    rank: int
    score: float
    content: str
    image_url: str | None = None  # URL to extracted diagram image


class RerankedChunkSnapshot(BaseModel):
    """Snapshot of a reranked chunk used to build the response payload."""

    chunk_id: str
    document_id: str = ""
    page: int | None = None
    rank: int
    score: float


class RAGState(BaseModel):
    # Multi-tenancy scaffold — resolved from auth context at API layer.
    # Every retrieval and ingestion operation should filter by tenant_id.
    tenant_id: str = ""

    original_query: str = ""
    query_rewrites: list[str] = Field(default_factory=list)
    rewritten_query: str = ""
    intent: QueryIntent = QueryIntent.SIMPLE
    decomposed_query: DecomposedQuery | None = None
    sub_queries: list[str] = Field(default_factory=list)
    retrieval_results: list[RetrievalResult] = Field(default_factory=list)
    reranked_results: list[RetrievalResult] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunkSnapshot] = Field(default_factory=list)
    reranked_chunks: list[RerankedChunkSnapshot] = Field(default_factory=list)
    context: str = ""
    supporting_chunks: list[str] = Field(default_factory=list)
    response: str = ""
    short_answer: str = ""
    citations: list[str] = Field(default_factory=list)
    faithfulness_score: float = 0.0
    answer_relevance_score: float = 0.0
    context_recall_score: float = 0.0
    confidence: float = 0.0
    faithfulness_passed: bool = False
    claim_count: int = 0
    supported_claims: int = 0

    # Operational / observability fields
    document_titles: dict[str, str] = Field(default_factory=dict)
    latency_ms: int = 0
    tokens_input: int = 0
    tokens_output: int = 0
