import asyncio

from src.graph.knowledge.graph_store import KnowledgeGraphStore
from src.observability.logging import get_logger
from src.shared.types import Chunk, RetrievalResult

logger = get_logger()


class KGRetriever:
    def __init__(self, graph_store: KnowledgeGraphStore) -> None:
        self.graph_store = graph_store

    async def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Retrieve relevant entities and their context from the knowledge graph.

        Uses entity extraction on the query to find matching graph nodes,
        then fetches their subgraph context.
        """
        from src.graph.knowledge.entity_extractor import extract_entities

        query_entities = await extract_entities(query)
        if not query_entities:
            return []

        # Fetch subgraphs for all entities in parallel
        async def _fetch_subgraph(entity_name: str) -> tuple[str, dict]:
            subgraph = await self.graph_store.get_subgraph(entity_name, depth=1)
            return entity_name, subgraph

        entity_names = [ent.get("name", "") for ent in query_entities[:top_k]]
        tasks = [_fetch_subgraph(name) for name in entity_names if name]
        subgraph_results = await asyncio.gather(*tasks, return_exceptions=True)

        results: list[RetrievalResult] = []
        for result in subgraph_results:
            if isinstance(result, Exception):
                logger.error("kg_subgraph_error", error=str(result))
                continue

            entity_name, subgraph = result
            context_parts = [f"Entity: {entity_name}"]
            for edge in subgraph.get("edges", []):
                context_parts.append(
                    f"{edge['source']} --[{edge['type']}]--> {edge['target']}"
                )

            if len(context_parts) > 1:
                chunk = Chunk(
                    id=f"kg_{entity_name}",
                    document_id="kg",
                    content="\n".join(context_parts),
                    section="Knowledge Graph",
                )
                results.append(RetrievalResult(chunk=chunk, score=1.0, source="kg"))

        logger.info("kg_retrieval", query=query[:80], results_count=len(results))
        return results[:top_k]
