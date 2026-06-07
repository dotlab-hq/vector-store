# PRD: Enterprise-Grade RAG Platform (10M+ Documents)

Stack

Python 3.12
LangGraph (orchestration)
LangChain (retrievers, embeddings, document loaders)
LangSmith (observability, evals, tracing)
DeepAgents (specialized retrieval/reasoning agents)
PostgreSQL + pgvector
Redis
Elasticsearch/OpenSearch
Neo4j (Knowledge Graph)
S3/MinIO
Kafka/NATS
FastAPI
Kubernetes

1. Product Vision

Build a production-grade RAG platform capable of:

10M+ documents
<2.5s response latency
Source-grounded answers
Hallucination detection
Multi-hop reasoning
Continuous self-improvement
Enterprise auditability
Multi-tenant support

The system should answer:

"What changed in policy X after regulation Y and how does it impact division Z?"

without hallucinating or losing citations.

2. High-Level Architecture
   User
   │
   ▼
   Query Understanding Agent
   │
   ▼
   Intent Router
   │
   ├── Simple Retrieval
   ├── Multi-Hop Retrieval
   ├── Analytical Reasoning
   └── KG Traversal
   │
   ▼
   Hybrid Retrieval Layer
   │
   ├── BM25
   ├── Dense
   ├── SPLADE
   └── Knowledge Graph
   │
   ▼
   Fusion Layer (RRF)
   │
   ▼
   Compression Layer
   │
   ▼
   Reranking Layer
   │
   ▼
   Context Builder
   │
   ▼
   Generation Agent
   │
   ▼
   Faithfulness Verification
   │
   ▼
   Citation Builder
   │
   ▼
   Response
3. Stage 1 — Ingestion Pipeline
   Goal

Convert raw enterprise documents into searchable knowledge.

Supported Sources
PDF
DOCX
PPTX
Wiki
Confluence
SharePoint
Emails
Databases
APIs
Git repositories
Processing Flow
Raw Document
↓
Normalization
↓
Deduplication
↓
Metadata Extraction
↓
Chunking
↓
Embedding
↓
Indexing
Metadata

Store:

{
"document_id": "...",
"title": "...",
"created_at": "...",
"updated_at": "...",
"author": "...",
"department": "...",
"entities": [...],
"version": "...",
"tags": [...]
} 4. Stage 2 — Chunking Service
Problem

Fixed token chunking breaks semantic meaning.

Semantic Chunking

Split on:

Heading
Paragraph
Section
Sentence boundaries

Never arbitrary token windows.

Parent Child Retrieval
Parent Chunk
512-1024 tokens

├─ Child A
├─ Child B
├─ Child C

Retrieve:

Child

Generate using:

Parent
Chunk Metadata
{
"chunk_id": "...",
"parent_id": "...",
"page": 17,
"position": 4,
"entities": [...],
"section": "Risk Analysis"
} 5. Stage 3 — Indexing Layer

Maintain multiple indexes.

BM25

Keyword matching.

Good for:

error code 0x8007
invoice number
SKU123
Dense Vector

Semantic retrieval.

Model examples:

bge-large
gte-large
e5-large
SPLADE

Learned sparse retrieval.

Benefits:

Better than BM25
Captures semantic expansion
Knowledge Graph

Neo4j.

Company
├─ owns
│
▼
Product
├─ regulated_by
▼
Policy

Supports relationship retrieval.

6. Stage 4 — Query Understanding Layer

LangGraph Agent.

Intent Classification

Classify:

Simple
MultiHop
Analytical
Comparative
Temporal
KGQuery
Query Rewriting

Example:

"What changed?"

becomes

"What changes occurred in Policy X between
2024 and 2025?"
HyDE

Generate hypothetical answer.

Question
↓
Hypothetical Answer
↓
Embedding
↓
Retrieval

Improves recall.

Query Decomposition

Example:

What changed after regulation Y and how
did it impact division Z?

Subqueries:

Q1 Regulation Y changes
Q2 Policy X changes
Q3 Impact on Division Z 7. Stage 5 — Retrieval Layer

Use multiple retrievers simultaneously.

DeepAgent Retrieval Team
Agent 1

Dense Retriever

Agent 2

BM25 Retriever

Agent 3

SPLADE Retriever

Agent 4

Knowledge Graph Retriever

Each runs independently.

Fusion

RRF

score += 1/(60 + rank)

Combines rankings.

8. Stage 6 — Compression Layer

Reduce noise.

LLMLingua

Remove irrelevant text.

10,000 tokens
↓
1,200 tokens

before reranking.

9. Stage 7 — Reranking Layer

Cross Encoder Models:

BGE-Reranker
Jina-Reranker
Cohere-Rerank

Input:

Query
Candidate Chunks

Output:

Relevance Scores

Top-K survives.

10. Stage 8 — Context Assembly

Build final prompt context.

Rules:

No duplicate chunks
Preserve chronology
Merge sibling chunks
Token budget enforcement 11. Stage 9 — Generation Layer

LangGraph Generation Agent.

Constraints:

Answer ONLY from provided context.
If evidence missing:
"I don't have sufficient evidence."

No external knowledge.

12. Stage 10 — Faithfulness Layer

Most RAG systems stop here.

We do not.

Claim Extraction
Answer
↓
Claims

Example:

Claim 1
Claim 2
Claim 3
NLI Verification

Models:

MiniCheck
TRUE
DeBERTa-NLI

Verify:

Claim
⊆
Retrieved Evidence
Failure Handling

If unsupported:

Regenerate

or

Abstain 13. Stage 11 — Citation Engine

Every statement linked.

[1] Policy.pdf p.17
[2] Memo.docx p.4
[3] Email.msg p.2

Clickable references.

14. Stage 12 — Memory Layer

Redis.

Semantic Cache
cosine(query,new_query) > 0.95

Return cached response.

Session Memory

Store:

Current task
User preferences
Conversation state 15. Stage 13 — Evaluation System

LangSmith.

Track:

Recall@K
Precision@K
MRR
NDCG
Faithfulness
Context Precision
Hallucination Rate
Online Evaluation

Sample:

5% production traffic

Run shadow pipeline.

Compare:

Pipeline A
vs
Pipeline B 16. Stage 14 — Observability

LangSmith + OpenTelemetry.

Track:

Retrieval latency
Rerank latency
Generation latency
Faithfulness failures
Missing citations
Cost per query 17. Stage 15 — Continuous Learning Flywheel
User Query
↓
Answer
↓
Evaluation
↓
Failure Detection
↓
Root Cause Analysis
↓
Retrieval Improvement
↓
Reindexing
↓
Better Answers 18. LangGraph Workflow
START
│
▼
Query Understanding
│
▼
Intent Router
│
├─ Dense
├─ BM25
├─ SPLADE
└─ KG
│
▼
Fusion
│
▼
Compression
│
▼
Rerank
│
▼
Context Build
│
▼
Generate
│
▼
Faithfulness Check
│
├─ PASS → Citations
│
└─ FAIL → Regenerate
│
▼
END
Success Metrics
Metric Target
Recall@20 >95%
Faithfulness >98%
Citation Coverage 100%
P95 Latency <2.5s
Hallucination Rate <1%
Availability 99.9%
Cache Hit Rate >40%
Cost Per Query <$0.01

This is roughly the architecture you'd expect for a modern enterprise RAG system rather than a basic "BM25 + vector + reranker" pipeline. The biggest differentiators are query understanding, multi-retriever fusion, faithfulness verification, and the feedback flywheel, which are often missing from most published RAG diagrams.

for vector store use faiss for now

Using FAISS initially is a very reasonable decision for a v1/v2 system.

For your architecture I'd modify the storage layer as follows:

Storage Layer
Document Storage
MinIO / S3
├── raw documents
├── normalized documents
└── versioned snapshots
Metadata Store
Postgres

documents
chunks
entities
versions
citations
retrieval_logs
evaluation_scores
Vector Store (V1)
FAISS

IndexHNSWFlat
or
IndexIVFPQ

Store:

{
"chunk_id": "...",
"embedding": [...],
"metadata_ref": "postgres_id"
}

Metadata should remain in PostgreSQL.

Never store all metadata inside FAISS.

Recommended FAISS Layout

For 10M+ chunks:

FAISS
├── Legal
├── Engineering
├── HR
├── Product
└── Finance

Domain sharding.

Avoid:

1 giant FAISS index

because retrieval latency grows and rebuilds become painful.

Retrieval Flow
User Query
│
▼
Intent Router
│
▼
Shard Selector
│
▼
FAISS Search
│
▼
Top 100
│
▼
Reranker
│
▼
Top 10
Embeddings

Recommended:

bge-large-en-v1.5
bge-m3
gte-large

For multi-language:

bge-m3
FAISS Index Choice
Development
faiss.IndexFlatIP

Pros:

Simple
Exact search

Cons:

Slow
Production
faiss.IndexHNSWFlat

Pros:

Fast
Excellent recall

Usually best first production choice.

Very Large Scale
faiss.IndexIVFPQ

Pros:

Smaller memory footprint
Faster

Cons:

Lower recall
More tuning
LangGraph Node Update
Query Understanding
│
▼
Retriever Coordinator
│
┌──────┼────────┐
▼ ▼ ▼
BM25 FAISS KG
│ │ │
└──────┴────────┘
│
▼
RRF Fusion
│
▼
Compression
│
▼
Reranking
│
▼
Generation
Future Migration Path

Design an abstraction from day one:

class VectorStore(ABC):

    @abstractmethod
    def search(self, query_embedding):
        pass

    @abstractmethod
    def insert(self, vectors):
        pass

Implement:

FaissVectorStore
QdrantVectorStore
WeaviateVectorStore
MilvusVectorStore
PgVectorStore

Then LangGraph never knows which backend is used.

What I'd Actually Build

Phase 1 (POC)

LangGraph
LangChain
FAISS
BM25
Redis
Postgres
FastAPI
LangSmith

Phase 2

Add:
SPLADE
Neo4j
HyDE
Query Decomposition
LLMLingua
Cross Encoder Reranker

Phase 3

Faithfulness Verification
RAGAS Online
Shadow Deployments
A/B Testing
Auto Re-indexing

Phase 4

Replace FAISS
→ Qdrant or Milvus

At ~10M+ chunks with frequent updates, distributed vector databases become much easier operationally than maintaining large FAISS indices, but FAISS is perfect for getting the entire system architecture working first.
