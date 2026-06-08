import time

from fastapi import APIRouter, HTTPException, Request

from apps.api.dependencies import get_workflow
from apps.api.schemas import (
    CitationItem as ApiCitationItem,
    EvaluationBlock,
    MetadataBlock,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
    RetrievalBlock,
    RerankedChunk,
    SourceItem as ApiSourceItem,
)
from src.config import settings
from src.database.repositories.document_repo import DocumentRepository
from src.database.session import async_session_factory
from src.generation.citations import CitationBuilder
from src.graph.state.schemas import RAGState
from src.observability.logging import get_logger

logger = get_logger()
router = APIRouter()


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


async def _build_document_info(
    reranked_results,
) -> tuple[dict[str, str], dict[str, str | None]]:
    """Look up document titles and s3_keys for all document_ids referenced in reranked results."""
    doc_ids = list(
        {r.chunk.document_id for r in reranked_results if r.chunk.document_id}
    )
    if not doc_ids:
        return {}, {}
    try:
        async with async_session_factory() as session:
            repo = DocumentRepository(session)
            docs = await repo.get_documents_by_ids(doc_ids)
            titles = {doc.id: doc.title for doc in docs.values()}
            s3_keys = {doc.id: doc.s3_key for doc in docs.values()}
            return titles, s3_keys
    except Exception as exc:
        logger.warning("doc_info_lookup_failed", error=str(exc))
        return {}, {}


@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest, http_request: Request) -> QueryResponse:
    workflow = get_workflow()
    initial_state = RAGState(original_query=request.query)

    start_ms = time.perf_counter()

    try:
        result = await workflow.ainvoke(initial_state)
    except Exception as exc:
        logger.error(
            "workflow_error",
            query=request.query[:200],
            error_type=type(exc).__name__,
            exc_info=True,
        )
        raise HTTPException(
            status_code=502,
            detail="Workflow execution failed. Please try again later.",
        ) from exc

    elapsed_ms = int((time.perf_counter() - start_ms) * 1000)

    # Handle both dict and RAGState return
    if isinstance(result, dict):
        state = RAGState(**result)
    else:
        state = result

    # Build structured citations & sources
    document_titles, document_s3_keys = await _build_document_info(
        state.reranked_results
    )
    citation_builder = CitationBuilder(document_titles)
    citations, sources = citation_builder.build(state.reranked_results)

    api_citations = [
        ApiCitationItem(
            id=c.id,
            source_id=c.source_id,
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            document_name=c.document_name,
            page=c.page,
            title=c.title,
            snippet=c.snippet,
            relevance_score=c.relevance_score,
        )
        for c in citations
    ]
    # Build base URL for presigned download links
    base_url = str(http_request.base_url).rstrip("/")

    api_sources = [
        ApiSourceItem(
            source_id=s.source_id,
            document_id=s.document_id,
            document_name=s.document_name,
            page=s.page,
            title=s.title,
            url=s.url,
            download_url=f"{base_url}/documents/{s.document_id}/download"
            if document_s3_keys.get(s.document_id)
            else None,
        )
        for s in sources
    ]

    # Assemble retrieval block
    retrieval_block = RetrievalBlock(
        query_rewrites=state.query_rewrites,
        retrieved_chunks=[
            RetrievedChunk(
                chunk_id=rc.chunk_id,
                document_id=rc.document_id,
                page=rc.page,
                rank=rc.rank,
                score=rc.score,
                content=rc.content,
            )
            for rc in state.retrieved_chunks
        ],
        reranked_chunks=[
            RerankedChunk(chunk_id=rc.chunk_id, rank=rc.rank, score=rc.score)
            for rc in state.reranked_chunks
        ],
        supporting_chunks=state.supporting_chunks,
    )

    # Assemble evaluation block
    evaluation_block = EvaluationBlock(
        confidence=state.confidence,
        faithfulness_score=state.faithfulness_score,
        faithfulness_passed=state.faithfulness_passed,
        answer_relevance_score=state.answer_relevance_score,
        context_recall_score=state.context_recall_score,
        claim_count=state.claim_count,
        supported_claims=state.supported_claims,
    )

    # Assemble metadata block (tokens come from generation node, latency is from this handler)
    tokens_in = state.tokens_input
    tokens_out = state.tokens_output
    if tokens_in == 0 and tokens_out == 0:
        # Fallback: estimate from response length if generation node didn't set it
        tokens_in = len(state.context) // 4
        tokens_out = len(state.response) // 4

    metadata_block = MetadataBlock(
        intent=state.intent.value,
        model=settings.openai_chat_model,
        embedding_model=settings.openai_embedding_model,
        reranker=RERANKER_MODEL,
        latency_ms=elapsed_ms,
        tokens_input=tokens_in,
        tokens_output=tokens_out,
    )

    return QueryResponse(
        query=state.original_query,
        answer=state.response,
        short_answer=state.short_answer,
        citations=api_citations,
        sources=api_sources,
        retrieval=retrieval_block,
        evaluation=evaluation_block,
        metadata=metadata_block,
    )
