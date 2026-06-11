"""Retriever protocol — single interface all retriever backends must satisfy.

Every retriever (HybridRetriever, KGRetriever, or any future backend)
conforms to this protocol so the rest of the system is backend-agnostic.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from src.shared.types import RetrievalResult


@runtime_checkable
class Retriever(Protocol):
    """Unified retriever interface used by the LangGraph pipeline."""

    async def retrieve(
        self, query: str, top_k: int = 20
    ) -> Sequence[RetrievalResult]: ...
