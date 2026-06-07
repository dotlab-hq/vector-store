import json

import numpy as np
import redis.asyncio as redis

from langchain_openai import OpenAIEmbeddings
from src.config import settings
from src.observability.logging import get_logger

logger = get_logger()

CACHE_TTL = 3600  # 1 hour
SIMILARITY_THRESHOLD = 0.95


class SemanticCache:
    def __init__(self, embedder: OpenAIEmbeddings) -> None:
        self._redis = redis.from_url(settings.redis_url, decode_responses=True)
        self.embedder = embedder
        self._keys: list[str] = []

    async def get(self, query: str) -> dict | None:
        query_embedding = np.array(await self.embedder.aembed_query(query))

        for key in self._keys:
            cached = await self._redis.get(key)
            if not cached:
                continue
            data = json.loads(cached)
            cached_embedding = np.array(data["embedding"])

            similarity = np.dot(query_embedding, cached_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(cached_embedding)
            )

            if similarity > SIMILARITY_THRESHOLD:
                logger.info("cache_hit", key=key, similarity=float(similarity))
                return data["response"]

        return None

    async def set(self, query: str, response: dict) -> None:
        embedding = await self.embedder.aembed_query(query)
        key = f"rag_cache:{hash(query)}"
        await self._redis.setex(
            key,
            CACHE_TTL,
            json.dumps({"query": query, "embedding": embedding, "response": response}),
        )
        self._keys.append(key)
        logger.info("cache_set", key=key)
