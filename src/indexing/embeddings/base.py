from abc import ABC, abstractmethod
from collections.abc import Sequence


class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]: ...
