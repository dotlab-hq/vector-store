from abc import ABC, abstractmethod
from collections.abc import Sequence

from src.shared.types import Chunk, RetrievalResult


class VectorStore(ABC):
    @abstractmethod
    async def search(
        self, query_embedding: list[float], top_k: int = 10
    ) -> Sequence[RetrievalResult]:
        ...

    @abstractmethod
    async def insert(self, chunks: Sequence[Chunk], embeddings: Sequence[list[float]]) -> None:
        ...

    @abstractmethod
    async def delete(self, chunk_ids: Sequence[str]) -> None:
        ...

    @abstractmethod
    async def count(self) -> int:
        ...
