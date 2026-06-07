# Architecture — Agentic RAG

## Overview

Enterprise-grade Retrieval-Augmented Generation platform targeting 10M+ documents with <2.5s P95 latency, hallucination detection, and citation-backed answers. Orchestrated by LangGraph, indexed via FAISS + BM25, verified by a faithfulness layer.

```
User Query
    │
    ▼
┌─────────────────────┐
│  Query Understanding │   Intent classification, rewriting, decomposition
│  (LLM Agent)         │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Intent Router       │   Simple | MultiHop | Analytical | Comparative | Temporal
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Hybrid Retrieval    │   Dense (FAISS)  ────  Sparse (BM25)
│  (Parallel)          │         │                  │
│                      │         └──────┬──────────┘
│                      │                ▼
│                      │         RRF Fusion (score += 1/(60+rank))
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Cross-Encoder       │   Rerank top-K for relevance
│  Reranker            │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Context Builder     │   Dedup, enforce token budget, preserve chronology
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Generation          │   LLM answers ONLY from provided context
│  (LLM Agent)         │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Faithfulness Check  │   Extract claims → verify against context
│  (NLI)               │        │
│                      │   ┌────┴────┐
│                      │   PASS     FAIL → Regenerate or Abstain
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Citation Builder    │   Link every statement to source chunk
└─────────┬───────────┘
          │
          ▼
     Final Response
```

---

## Design Principles

### 1. Adapters Over Direct Dependencies

Every external service is wrapped in an adapter interface:

- `VectorStore(ABC)` → `FaissVectorStore` (swap for Qdrant/Milvus later)
- `EmbeddingProvider(ABC)` → `OpenAIEmbeddings` (swap for local models)
- `DocumentLoader(ABC)` → `PdfLoader | DocxLoader | TextLoader`

No business logic calls an external SDK directly. This makes the system testable and the vector store swappable without touching the retrieval pipeline.

### 2. Pydantic at Every Boundary

All state, schema, and configuration uses Pydantic:
- `Settings` for env config (pydantic-settings)
- `RAGState` for the LangGraph workflow state
- `Chunk`, `Document`, `RetrievalResult` at data layer
- `QueryClassification`, `DecomposedQuery` at agent layer
- `QueryRequest`, `QueryResponse` at API layer

This ensures validation, serialization, and type safety at every boundary.

### 3. LangGraph as the Orchestrator

The full RAG pipeline is a StateGraph with typed state, not a procedural script. Each step is a pure function that receives `RAGState` and returns a partial update:

```
START → query_understanding → retrieval → reranking → context_building → generation → faithfulness_check → END
```

Advantages over linear code:
- Easy to add/remove nodes (e.g., inject a safety filter)
- Conditional routing per intent (different retrieval strategies per query type)
- Built-in LangSmith tracing for every step
- State is inspectable at any point

### 4. Structured Logging Over Print

All logging uses structlog with JSON output. Every log line carries:
- `query_id`, `user_id`, `workflow_id`, `node_name`, `latency_ms`

This feeds directly into LangSmith and OpenTelemetry.

---

## Process Flows

### A. Ingestion Pipeline

```
File → DocumentLoader.load()
     → RawDocument (content + metadata)
     → MetadataExtractor.extract()
         → Document Pydantic model (id, title, author, tags, entities)
     → stored in PostgreSQL via DocumentRepository
     → ParentChildChunker.build()
         → Parent chunks (1024 tokens) + Child chunks (256 tokens)
     → stored as ChunkModel rows in PostgreSQL
```

**Design choices:**
- Parent-child chunking enables "retrieve child, generate with parent" context — better recall than flat chunks
- Metadata in PostgreSQL, not in the vector index — FAISS stores only the embedding and a reference pointer
- Batch-sized embedding + indexing via the `Indexer` for memory control at scale

### B. Indexing Pipeline

```
Chunk texts → EmbeddingProvider.embed() → embeddings list
    │
    ├──→ FaissVectorStore.insert() → FAISS IndexFlatIP (normalized cosine)
    │
    └──→ Bm25Store.insert() → BM25Okapi corpus
```

**Design choices:**
- FAISS IndexFlatIP with L2 normalization gives cosine similarity in inner-product form
- BM25 provides keyword-based retrieval for exact terms, error codes, SKU lookups that dense embeddings miss
- Dual indexing enables hybrid fusion: dense finds semantic matches, BM25 catches lexical matches

### C. Query Understanding

```
User Query
    │
    ├──→ LLM classifies intent: Simple | MultiHop | Analytical | Comparative | Temporal | KG_Query
    │
    ├──→ LLM rewrites query to be more specific and search-friendly
    │
    └──→ If complex: LLM decomposes into independent sub-queries
```

**Design choices:**
- Intent classification enables different retrieval strategies per query type (e.g., KG traversal for relationship queries)
- Query rewriting improves recall — "what changed" → "what changes occurred in Policy X between 2024 and 2025"
- Decomposition converts multi-hop questions into sequential sub-queries, each answered independently

### D. Hybrid Retrieval

```
Query → DenseRetriever (FAISS)
     → SparseRetriever (BM25)
     → RRF Fusion: score += 1/(60 + rank) for each list
     → Top-K by fused score
```

**Design choices:**
- Dense + sparse combined outperforms either alone (proven by every TREC benchmark)
- RRF is parameter-free (k=60 is the standard) and doesn't require score normalization between different retriever types
- Each retriever runs independently — trivially parallelizable

### E. Generation

```
Reranked chunks → ContextBuilder.trim_to_token_budget()
                → Assembly with [Source: chunk_id] markers
                → LLM prompted: "Answer ONLY from provided context"
                → Response with inline citations
```

**Design choices:**
- Token budget enforcement prevents context overflow (8K token limit)
- Source markers in context make citation extraction deterministic
- Constrained prompt ("answer only from context") reduces hallucination risk

### F. Faithfulness Verification

```
Response → ClaimExtraction (LLM) → list of atomic claims
         → For each claim: verify against retrieved context
         → Score = supported_claims / total_claims
         → If score < 0.8 → regenerate or abstain
```

**Design choices:**
- Two-stage: extract then verify, not a single "is this faithful?" judgement
- Score threshold at 0.8 balances recall and precision
- The system can regenerate with different context rather than returning an unfaithful answer

### G. Memory & Caching

```
Semantic Cache (Redis):
  Query → embed → cosine similarity against cached queries
  If >0.95 → return cached response (1-hour TTL)

Session Memory (Redis):
  Session ID → ordered message list (30-minute TTL)
  Context window management for multi-turn conversations
```

**Design choices:**
- Semantic cache catches near-identical queries without exact matching
- Redis for both because it's fast, simple, and available at any scale

---

## Module Structure

```
src/
├── config/           Pydantic Settings (env vars, DB URLs, model names)
├── shared/           Protocols + shared Pydantic types
├── database/         SQLAlchemy async models + repositories
├── observability/    structlog configuration
├── llm/              LLM adapter (ChatOpenAI)
├── ingestion/        Loaders, chunkers, metadata extraction
├── indexing/         Embeddings, FAISS store, BM25 store
├── agents/           Intelligent agents (query understanding)
├── retrieval/        Dense, sparse, hybrid fusion, reranking
├── graph/            LangGraph state, nodes, routers, workflow
├── generation/       Context building, citations, faithfulness
└── memory/           Semantic cache + session store

apps/
└── api/              FastAPI entrypoints (/query, /ingest, /health)
```

---

## Data Flow (Request Lifecycle)

```
POST /query
    │
    ├── 1. FastAPI receives the request
    ├── 2. Extracts original_query into RAGState
    ├── 3. StateGraph.ainvoke(RAGState) starts the pipeline
    │
    │   Node 1: query_understanding
    │     - Classify intent (LLM)
    │     - Rewrite query (LLM)
    │     - Decompose if complex (LLM)
    │     → Updated state: intent, rewritten_query, sub_queries
    │
    │   Node 2: retrieval
    │     - For each sub_query: dense (FAISS) + sparse (BM25) via HybridRetriever
    │     - RRF fusion across all result lists
    │     → Updated state: retrieval_results
    │
    │   Node 3: reranking
    │     - Cross-encoder scores (query, chunk) for each result
    │     - Keep top-K by relevance score
    │     → Updated state: reranked_results
    │
    │   Node 4: context_building
    │     - Deduplicate chunks
    │     - Build context string with source markers
    │     - Enforce token budget
    │     → Updated state: context, citations
    │
    │   Node 5: generation
    │     - LLM generates answer from context only
    │     → Updated state: response
    │
    │   Node 6: faithfulness_check
    │     - Extract claims from response (LLM)
    │     - Verify each claim against context (LLM)
    │     → Updated state: faithfulness_score, faithfulness_passed
    │
    │   Router: faithfulness_router
    │     - If PASS → END
    │     - If FAIL → (planned: regenerate with different context or abstain)
    │
    └── 4. Response assembled with query, answer, citations, faithfulness_score
```

---

## Merits & Trade-offs

### Strengths

| Aspect | Merits |
|---|---|
| **Correctness** | Faithfulness verification after generation catches hallucinations. Every answer has citations. |
| **Traceability** | LangGraph + LangSmith trace every step. Structured logging carries query_id through the entire pipeline. |
| **Recall** | Hybrid retrieval (dense + sparse) + RRF fusion captures both semantic and lexical matches. Query rewriting improves recall further. |
| **Maintainability** | Adapter pattern means swapping FAISS for Qdrant, or OpenAI for a local model, touches exactly one file. Clean dependency direction (apps → graph → agents → retrieval → database). |
| **Testability** | Every node is a pure function on RAGState. Loaders, chunkers, and stores have clear interfaces. |
| **Security** | No prompts in business logic (all in `src/generation/prompts/`). No raw `os.environ`. Pydantic validation at all boundaries. |
| **Multi-tenancy** | Domain-sharded FAISS indexes. Department/tenant tags in metadata. |

### Design Trade-offs

| Decision | Trade-off |
|---|---|
| **FAISS over Qdrant** | FAISS is simpler to set up (no running server) but harder to scale to 10M+ with frequent updates. We accept an operational migration to Qdrant/Milvus at scale. |
| **LLM-based faithfulness** | Using an LLM for claim extraction + verification is more accurate than heuristic NLI but adds latency (~500ms) and cost per query. We accept this for correctness over speed. |
| **Parent-child chunking** | More storage and index overhead than flat chunking, but significantly better retrieval quality because the generator sees the full parent context while the retriever matches on specific child passages. |
| **LLM agents (not rule-based)** | Intent classification and query rewriting via LLM is slower and more expensive than a regex router. We accept the cost for flexibility: an LLM handles novel query patterns without code changes. |
| **Synchronous cross-encoder** | The reranker's CrossEncoder runs synchronously in `predict()`. Using `run_in_executor` avoids event-loop blocking but doesn't solve the per-query latency. We accept ~100ms per rerank for the quality gain over pure embedding similarity. |
| **PostgreSQL + asyncpg** | Full ACID compliance with async access. Slower than a dedicated vector DB for similarity search, but metadata queries, analytics, and joins benefit from relational integrity. |
| **Redis for semantic cache** | Using Redis OSS (without RediSearch) means cache scans are O(N) over stored embeddings. This is acceptable for hundreds of entries but doesn't scale to millions without a migration to Redis Stack. |

### Performance Targets

| Layer | Budget |
|---|---|
| Embedding + FAISS search | < 50 ms |
| BM25 search | < 10 ms |
| RRF fusion | < 20 ms |
| Cross-encoder reranking | < 100 ms |
| Generation (LLM) | < 2 s |
| Faithfulness check | < 500 ms |
| **Total P95** | **< 2.5 s** |

---

## Future Migrations

1. **FAISS → Qdrant/Milvus** at ~10M chunks — distributed vector search without index rebuilds
2. **Redis → Redis Stack** for native vector similarity search in the semantic cache
3. **SPLADE** as a third retriever for learned sparse retrieval (better than BM25)
4. **Neo4j** knowledge graph for entity-relationship queries
5. **LLMLingua** for prompt compression before the cross-encoder (reduce cost and latency)
6. **Online evaluation** with 5% production traffic via shadow pipeline — RAGAS metrics tracked in LangSmith
7. **Continuous learning flywheel** — failed queries trigger reindexing of relevant documents

---

## Development

```bash
# Dependencies
uv sync

# Environment
# Create .env with:
#   OPENAI_API_KEY=sk-...
#   DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/agentic_rag
#   REDIS_URL=redis://localhost:6379/0

# Run API
uvicorn apps.api.main:app --reload

# CLI entrypoint
uv run python -m src.main
```
