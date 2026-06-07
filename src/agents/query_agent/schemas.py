from pydantic import BaseModel, Field

from src.shared.types import QueryIntent


class QueryClassification(BaseModel):
    intent: QueryIntent
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


class RewrittenQuery(BaseModel):
    original: str
    rewritten: str
    additions: str = ""


class DecomposedQuery(BaseModel):
    original: str
    sub_queries: list[str] = Field(default_factory=list)
