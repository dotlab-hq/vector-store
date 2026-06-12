from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class QueryIntent(str, Enum):
    SIMPLE = "simple"
    MULTI_HOP = "multi_hop"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    TEMPORAL = "temporal"
    KG_QUERY = "kg_query"


class Document(BaseModel):
    id: str
    title: str
    source_path: str
    source_type: str  # pdf, docx, txt, etc.
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    author: str = ""
    department: str = ""
    tags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    content_text: str = ""  # raw text — used by async worker to re-chunk


class Chunk(BaseModel):
    id: str
    document_id: str
    content: str
    parent_id: str | None = None
    page_number: int | None = None
    position: int = 0
    section: str = ""
    entities: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    vector_store_id: str | None = None
    attributes: dict = Field(default_factory=dict)
    image_url: str | None = None  # S3 URL for extracted diagrams/images


class RetrievalResult(BaseModel):
    chunk: Chunk
    score: float
    source: str  # "dense", "sparse", "bm25", etc.


class QueryResult(BaseModel):
    query: str
    intent: QueryIntent
    rewritten_query: str = ""
    results: list[RetrievalResult] = Field(default_factory=list)
    response: str = ""
    citations: list[str] = Field(default_factory=list)
    faithfulness_score: float = 0.0
