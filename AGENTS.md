# AGENTS.md

# Enterprise RAG Platform

This document defines the architecture, coding standards, folder structure, and agent rules for all AI coding agents working in this repository.

---

# Project Goal

Build a production-grade Retrieval Augmented Generation (RAG) platform capable of:

* 10M+ documents
* Multi-hop reasoning
* Citation-backed responses
* Hallucination detection
* Knowledge Graph retrieval
* Continuous evaluation
* LangGraph orchestration
* LangSmith observability

The system must prioritize:

1. Correctness
2. Traceability
3. Reliability
4. Maintainability
5. Performance

Never optimize for speed at the cost of answer quality.

---

# Technology Stack

## Core

* Python 3.12+
* LangGraph
* LangChain
* LangSmith
* DeepAgents
* FastAPI

## Storage

* PostgreSQL
* Redis
* FAISS (initially)
* MinIO / S3

## Search

* BM25
* Dense Retrieval
* SPLADE (future)

## Evaluation

* RAGAS
* LangSmith Evals

---

# Repository Structure

```text
rag-platform/

├── apps/
│   ├── api/
│   ├── worker/
│   └── dashboard/
│
├── src/
│
│   ├── ingestion/
│   │   ├── loaders/
│   │   ├── parsers/
│   │   ├── chunking/
│   │   ├── metadata/
│   │   └── pipelines/
│   │
│   ├── indexing/
│   │   ├── embeddings/
│   │   ├── faiss/
│   │   ├── bm25/
│   │   └── synchronization/
│   │
│   ├── retrieval/
│   │   ├── dense/
│   │   ├── sparse/
│   │   ├── hybrid/
│   │   ├── fusion/
│   │   └── reranking/
│   │
│   ├── graph/
│   │   ├── workflows/
│   │   ├── nodes/
│   │   ├── state/
│   │   └── routers/
│   │
│   ├── agents/
│   │   ├── query_agent/
│   │   ├── retrieval_agent/
│   │   ├── generation_agent/
│   │   └── evaluation_agent/
│   │
│   ├── generation/
│   │   ├── prompts/
│   │   ├── context/
│   │   ├── citations/
│   │   └── faithfulness/
│   │
│   ├── memory/
│   │   ├── cache/
│   │   ├── session/
│   │   └── semantic/
│   │
│   ├── evaluation/
│   │   ├── ragas/
│   │   ├── benchmarks/
│   │   └── metrics/
│   │
│   ├── observability/
│   │   ├── tracing/
│   │   ├── logging/
│   │   └── telemetry/
│   │
│   ├── database/
│   │   ├── models/
│   │   ├── repositories/
│   │   └── migrations/
│   │
│   ├── config/
│   └── shared/
│
├── tests/
│
├── scripts/
│
├── docs/
│
└── AGENTS.md
```

---

# Architecture Principles

## Rule 1

Business logic belongs in:

```text
src/
```

Never place business logic inside:

```text
apps/
```

Apps should contain entrypoints only.

---

## Rule 2

All external services must have adapters.

Bad:

```python
openai.chat.completions.create(...)
```

Good:

```python
llm_provider.generate(...)
```

---

## Rule 3

Never directly access FAISS.

Always use:

```python
VectorStore
```

interface.

Example:

```python
class VectorStore(Protocol):
    def search(...)
    def insert(...)
    def delete(...)
```

---

## Rule 4

Every retrieval operation must be traceable.

Must log:

* query
* retrieved chunks
* scores
* reranked scores
* final context

---

## Rule 5

No hidden prompts.

Prompts belong only in:

```text
src/generation/prompts/
```

---

# LangGraph Rules

All workflows belong under:

```text
src/graph/workflows/
```

Node implementations belong under:

```text
src/graph/nodes/
```

Never create workflow logic inside nodes.

Nodes should be pure functions.

Example:

```python
def retrieve_node(state):
    ...
    return updated_state
```

---

# Agent Rules

Every agent must have:

```python
agent.py
prompts.py
schemas.py
tests.py
```

Example:

```text
agents/
└── retrieval_agent/
    ├── agent.py
    ├── prompts.py
    ├── schemas.py
    └── tests.py
```

---

# State Management

All graph state definitions belong in:

```text
src/graph/state/
```

State must use:

```python
Pydantic
```

Never use untyped dictionaries.

---

# Configuration Rules

Environment variables must be accessed through:

```python
Settings
```

class.

Never call:

```python
os.getenv()
```

inside business logic.

---

# Dependency Rules

Allowed Direction:

```text
apps
 ↓
graph
 ↓
agents
 ↓
retrieval
 ↓
database
```

Never import upward.

Bad:

```python
database -> graph
```

Good:

```python
graph -> database
```

---

# Testing Rules

Every feature requires:

## Unit Tests

Location:

```text
tests/unit/
```

## Integration Tests

Location:

```text
tests/integration/
```

## Evaluation Tests

Location:

```text
tests/evals/
```

---

# Logging Rules

Use structured logging only.

Required fields:

```python
query_id
user_id
workflow_id
node_name
latency_ms
```

Never use print().

---

# Performance Targets

Dense Retrieval:
< 50 ms

Fusion:
< 20 ms

Reranking:
< 100 ms

Generation:
< 2 seconds

Faithfulness Check:
< 500 ms

Total P95:
< 2.5 seconds

---

# Code Style

Mandatory:

* Type hints everywhere
* Pydantic models
* Dataclasses when appropriate
* Async IO for network operations
* Ruff
* Pyright
* Black

---

# Documentation

Every public function requires:

```python
"""
Purpose

Parameters

Returns

Raises
"""
```

---

# Forbidden

Do not:

* Create circular imports
* Access databases directly from agents
* Put prompts inline
* Use global mutable state
* Store secrets in code
* Bypass repositories
* Bypass vector store abstraction

---

# Definition of Done

A task is complete only if:

* Code compiles
* Tests pass
* Type checking passes
* Linting passes
* LangSmith traces work
* Documentation updated
* Metrics emitted
* No hardcoded secrets

If any item is missing, the task is not complete.
