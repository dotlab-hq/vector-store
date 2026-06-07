from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ParserSignal:
    name: str
    content: str
    confidence: float
    source: str
    metadata: dict[str, Any]


class MediaParser(ABC):
    @abstractmethod
    def can_parse(self, file_path: Path) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def parse(self, file_path: Path, file_bytes: bytes, fallback_text: str = "") -> ParserSignal | None:
        raise NotImplementedError
