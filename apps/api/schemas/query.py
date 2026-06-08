from typing import Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)


class CitationItem(BaseModel):
    """A single citation pointing at a specific chunk."""

    id: int
    source_id: str
    chunk_id: str
    document_id: str
    document_name: str
    page: Optional[int] = None
    title: str = ""
    snippet: str = ""
    relevance_score: float = 0.0


class SourceItem(BaseModel):
    """A deduplicated document-level source referenced by one or more citations."""

    source_id: str
    document_id: str
    document_name: str
    page: Optional[int] = None
    title: str = ""
    url: Optional[str] = None
    download_url: Optional[str] = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    page: Optional[int] = None
    rank: int
    score: float
    content: str


class RerankedChunk(BaseModel):
    chunk_id: str
    rank: int
    score: float


class RetrievalBlock(BaseModel):
    query_rewrites: list[str] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    reranked_chunks: list[RerankedChunk] = Field(default_factory=list)
    supporting_chunks: list[str] = Field(default_factory=list)


class EvaluationBlock(BaseModel):
    confidence: float = 0.0
    faithfulness_score: float = 0.0
    faithfulness_passed: bool = False
    answer_relevance_score: float = 0.0
    context_recall_score: float = 0.0
    claim_count: int = 0
    supported_claims: int = 0


class MetadataBlock(BaseModel):
    intent: str = ""
    model: str = ""
    embedding_model: str = ""
    reranker: str = ""
    latency_ms: int = 0
    tokens_input: int = 0
    tokens_output: int = 0


class QueryResponse(BaseModel):
    query: str
    answer: str
    short_answer: str = ""
    citations: list[CitationItem] = Field(default_factory=list)
    sources: list[SourceItem] = Field(default_factory=list)
    retrieval: RetrievalBlock = Field(default_factory=RetrievalBlock)
    evaluation: EvaluationBlock = Field(default_factory=EvaluationBlock)
    metadata: MetadataBlock = Field(default_factory=MetadataBlock)


class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=1)
    title: str = "Untitled"


# Keep IngestRequest as alias for backward compatibility
IngestRequest = IngestTextRequest


class IngestResponse(BaseModel):
    document_id: str
    title: str
    chunks_created: int = 0
    processing_status: str = "pending"
