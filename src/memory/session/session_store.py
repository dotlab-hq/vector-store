import json
from datetime import datetime

import redis.asyncio as redis

from src.config import settings
from src.observability.logging import get_logger

logger = get_logger()

SESSION_TTL = 1800  # 30 minutes


class SessionStore:
    def __init__(self) -> None:
        self._redis = redis.from_url(settings.redis_url, decode_responses=True)

    async def get_messages(self, session_id: str) -> list[dict]:
        key = f"session:{session_id}"
        data = await self._redis.get(key)
        if data:
            return json.loads(data)
        return []

    async def add_message(self, session_id: str, role: str, content: str) -> None:
        key = f"session:{session_id}"
        messages = await self.get_messages(session_id)
        messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        })
        await self._redis.setex(key, SESSION_TTL, json.dumps(messages))
        logger.info("session_message_added", session_id=session_id, role=role)

    async def clear(self, session_id: str) -> None:
        key = f"session:{session_id}"
        await self._redis.delete(key)
