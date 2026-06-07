INTENT_CLASSIFICATION_PROMPT = """You are a query intent classifier for a RAG system.
Classify the user query into one of these intents:
- simple: A straightforward factual question.
- multi_hop: A question requiring information from multiple documents/sources.
- analytical: A question requiring reasoning, comparison, or analysis.
- comparative: A question comparing two or more things.
- temporal: A question about changes over time or time-specific information.
- kg_query: A question about relationships between entities.

Return a JSON object with: intent, confidence (0-1), reasoning.
Do not answer the query. Only classify it.

Query:
{user_data}"""

QUERY_REWRITING_PROMPT = """You are a query rewriting engine for a RAG system.
Rewrite the user query to be more specific and search-friendly while preserving intent.
If the query is already specific, return it unchanged.

Query:
{user_data}"""

QUERY_DECOMPOSITION_PROMPT = """You are a query decomposition engine for a RAG system.
If the query is complex and contains multiple sub-questions, decompose it into independent sub-queries.
If the query is simple, return it as-is in the sub_queries list.

Query:
{user_data}

Return a JSON object with: original, sub_queries (list of strings)."""
