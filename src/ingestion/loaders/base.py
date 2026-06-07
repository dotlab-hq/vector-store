"""Base loader abstractions used by all document loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel


class RawDocument(BaseModel):
    """Raw document output from a loader, prior to chunking/metadata extraction."""

    content: str
    metadata: dict
    source_path: str


class DocumentLoader(ABC):
    """Abstract interface for document loaders."""

    @abstractmethod
    def can_load(self, file_path: Path) -> bool:
        """Return True if this loader can handle the given file."""
        ...

    @abstractmethod
    def load(self, file_path: Path) -> RawDocument:
        """Load and extract text content from the given file."""
        ...
