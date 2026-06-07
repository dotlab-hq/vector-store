import asyncio

from src.graph.knowledge.entity_extractor import extract_entities
from src.graph.knowledge.neo4j_client import Neo4jClient
from src.graph.knowledge.relationship_extractor import extract_relationships
from src.observability.logging import get_logger

logger = get_logger()

_KG_CONCURRENCY = 10  # max chunks processed concurrently


class KnowledgeGraphStore:
    def __init__(self, neo4j: Neo4jClient) -> None:
        self.neo4j = neo4j

    async def ingest_chunk(self, chunk_id: str, content: str, metadata: dict | None = None) -> dict:
        """Extract entities and relationships from a chunk, store in Neo4j."""
        entities = await extract_entities(content)
        relationships = await extract_relationships(content, entities)

        # Store entities
        for ent in entities:
            name = ent.get("name", "")
            ent_type = ent.get("type", "Unknown")
            props = {"chunk_id": chunk_id}
            if metadata:
                props.update(metadata)
            await self.neo4j.create_entity(name, ent_type, props)

        # Store relationships
        for rel in relationships:
            src = rel.get("source", "")
            tgt = rel.get("target", "")
            rel_type = rel.get("relationship", "RELATED_TO")
            # Convert to safe Cypher relationship type (uppercase, underscores)
            safe_type = rel_type.upper().replace(" ", "_").replace("-", "_")
            await self.neo4j.create_relationship(src, tgt, safe_type, {"chunk_id": chunk_id})

        logger.info(
            "chunk_ingested_to_kg",
            chunk_id=chunk_id,
            entities=len(entities),
            relationships=len(relationships),
        )

        return {
            "entities": len(entities),
            "relationships": len(relationships),
        }

    async def ingest_document(self, document_id: str, chunks: list[dict]) -> dict:
        """Ingest all chunks from a document into the knowledge graph concurrently."""
        total_entities = 0
        total_relationships = 0
        sem = asyncio.Semaphore(_KG_CONCURRENCY)

        async def _process(chunk: dict) -> dict:
            async with sem:
                return await self.ingest_chunk(
                    chunk_id=chunk.get("id", ""),
                    content=chunk.get("content", ""),
                    metadata={"document_id": document_id},
                )

        results = await asyncio.gather(*[_process(c) for c in chunks])
        for r in results:
            total_entities += r["entities"]
            total_relationships += r["relationships"]

        logger.info(
            "document_ingested_to_kg",
            document_id=document_id,
            total_chunks=len(chunks),
            total_entities=total_entities,
            total_relationships=total_relationships,
        )

        return {
            "total_entities": total_entities,
            "total_relationships": total_relationships,
        }

    async def query_entity(self, entity_name: str) -> dict | None:
        """Get entity and its relationships from the graph."""
        entity = await self.neo4j.get_entity(entity_name)
        if not entity:
            return None

        relationships = await self.neo4j.get_entity_relationships(entity_name)
        return {
            "entity": entity,
            "related_entities": relationships,
        }

    async def get_subgraph(self, entity_name: str, depth: int = 2) -> dict:
        """Get a subgraph for traversal-based retrieval."""
        return await self.neo4j.get_subgraph(entity_name, depth=depth)
