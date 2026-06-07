import re

from neo4j import AsyncDriver, AsyncGraphDatabase

from src.config import settings
from src.observability.logging import get_logger

logger = get_logger()

# Allowlist for relationship types to prevent Cypher injection
_VALID_REL_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")
_MAX_DEPTH = 10  # Maximum graph traversal depth


def _sanitize_relationship_type(rel_type: str) -> str:
    """Sanitize relationship type to prevent Cypher injection.

    Only allows uppercase alphanumeric + underscores.
    """
    safe = rel_type.upper().replace(" ", "_").replace("-", "_")
    safe = re.sub(r"[^A-Z0-9_]", "", safe)
    if not _VALID_REL_PATTERN.match(safe):
        raise ValueError(f"Invalid relationship type: {rel_type!r}")
    return safe


def _validate_depth(depth: int) -> int:
    """Clamp depth to safe bounds."""
    return max(1, min(depth, _MAX_DEPTH))


class Neo4jClient:
    def __init__(self, uri: str = "", user: str = "", password: str = "") -> None:
        self._driver: AsyncDriver = AsyncGraphDatabase.driver(
            uri or settings.neo4j_uri,
            auth=(user or settings.neo4j_user, password or settings.neo4j_password),
        )

    async def verify_connectivity(self) -> bool:
        try:
            await self._driver.verify_connectivity()
            return True
        except Exception as e:
            logger.error("neo4j_connectivity_failed", error=str(e))
            return False

    async def run_query(self, query: str, parameters: dict | None = None) -> list[dict]:
        """Run a Cypher query and return results as list of dicts."""
        async with self._driver.session() as session:
            result = await session.run(query, parameters or {})
            records = await result.data()
            return records

    async def create_entity(self, name: str, entity_type: str, properties: dict | None = None) -> dict:
        """Create or merge an entity node."""
        query = """
        MERGE (e:Entity {name: $name})
        SET e.type = $entity_type
        SET e += $properties
        RETURN e {.*}
        """
        params = {"name": name, "entity_type": entity_type, "properties": properties or {}}
        results = await self.run_query(query, params)
        return results[0] if results else {}

    async def create_relationship(
        self,
        source_name: str,
        target_name: str,
        relationship_type: str,
        properties: dict | None = None,
    ) -> dict:
        """Create a relationship between two entities."""
        safe_type = _sanitize_relationship_type(relationship_type)
        query = f"""
        MERGE (s:Entity {{name: $source_name}})
        MERGE (t:Entity {{name: $target_name}})
        MERGE (s)-[r:{safe_type}]->(t)
        SET r += $properties
        RETURN type(r) AS rel_type, s.name AS source, t.name AS target
        """
        params = {
            "source_name": source_name,
            "target_name": target_name,
            "properties": properties or {},
        }
        results = await self.run_query(query, params)
        return results[0] if results else {}

    async def get_entity(self, name: str) -> dict | None:
        """Get an entity by name."""
        query = "MATCH (e:Entity {name: $name}) RETURN e {.*}"
        results = await self.run_query(query, {"name": name})
        return results[0].get("e") if results else None

    async def get_entity_relationships(self, name: str, depth: int = 2) -> list[dict]:
        """Get all relationships for an entity up to a given depth."""
        safe_depth = _validate_depth(depth)
        query = """
        MATCH (e:Entity {name: $name})-[r*1..$depth]-(related:Entity)
        RETURN DISTINCT related.name AS name, related.type AS type
        """
        return await self.run_query(query, {"name": name, "depth": safe_depth})

    async def get_subgraph(self, entity_name: str, depth: int = 2) -> dict:
        """Get a subgraph around an entity for visualization/querying."""
        safe_depth = _validate_depth(depth)
        query = """
        MATCH path = (e:Entity {name: $name})-[*1..$depth]-(related:Entity)
        RETURN path
        """
        results = await self.run_query(query, {"name": entity_name, "depth": safe_depth})

        nodes: dict[str, dict] = {}
        edges: list[dict] = []

        for record in results:
            path = record.get("path")
            if path is None:
                continue
            for node in path.nodes:
                name = node.get("name", "")
                if name not in nodes:
                    nodes[name] = {"name": name, "type": node.get("type", "Unknown")}
            for rel in path.relationships:
                start_name = rel.start_node.get("name", "")
                end_name = rel.end_node.get("name", "")
                edges.append({
                    "source": start_name,
                    "target": end_name,
                    "type": rel.type,
                })

        return {"nodes": list(nodes.values()), "edges": edges}

    async def close(self) -> None:
        await self._driver.close()
