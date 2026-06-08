from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import mimetypes
from enum import Enum
from pathlib import Path
from typing import Any

from src.ingestion.parsers import (
    AudioParser,
    DocumentParser,
    ImageParser,
    MediaParser,
    ParserSignal,
    VideoParser,
)


class MediaType(str, Enum):
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"
    TEXT = "text"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class MediaSignal(ParserSignal):
    pass


@dataclass(slots=True)
class MediaSignalScore:
    signal_name: str
    coverage: float
    specificity: float
    confidence: float
    weighted_score: float


@dataclass(slots=True)
class MediaExtractionResult:
    media_type: MediaType
    summary: str
    signals: list[MediaSignal]
    scores: list[MediaSignalScore]
    best_signal: MediaSignal | None
    enriched_content: str
    metadata: dict[str, Any]


class MediaProcessingService:
    """Collect the richest possible representation of a media asset."""

    def __init__(self) -> None:
        self.parsers: list[MediaParser] = [
            DocumentParser(),
            ImageParser(),
            AudioParser(),
            VideoParser(),
        ]

    def detect_media_type(self, file_path: Path) -> MediaType:
        suffix = file_path.suffix.lower()
        mime_type, _ = mimetypes.guess_type(str(file_path))
        mime_type = (mime_type or "").lower()
        if mime_type.startswith("image/") or suffix in {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".bmp",
            ".tiff",
            ".gif",
        }:
            return MediaType.IMAGE
        if mime_type.startswith("audio/") or suffix in {
            ".mp3",
            ".wav",
            ".m4a",
            ".ogg",
            ".flac",
        }:
            return MediaType.AUDIO
        if mime_type.startswith("video/") or suffix in {
            ".mp4",
            ".mov",
            ".mkv",
            ".webm",
            ".avi",
        }:
            return MediaType.VIDEO
        if suffix in {
            ".pdf",
            ".docx",
            ".doc",
            ".pptx",
            ".ppt",
            ".txt",
            ".md",
            ".html",
            ".htm",
            ".csv",
            ".json",
            ".xml",
        }:
            return MediaType.DOCUMENT
        if suffix in {".txt", ".md"}:
            return MediaType.TEXT
        return MediaType.UNKNOWN

    async def process(
        self, file_path: Path, *, fallback_text: str = ""
    ) -> MediaExtractionResult:
        media_type = self.detect_media_type(file_path)
        file_bytes = await asyncio.to_thread(file_path.read_bytes)
        tasks = [
            asyncio.create_task(parser.parse(file_path, file_bytes, fallback_text))
            for parser in self.parsers
            if parser.can_parse(file_path)
        ]
        raw_signals = [
            signal for signal in await asyncio.gather(*tasks) if signal is not None
        ]
        scored_signals = [
            self._score_signal(signal, fallback_text) for signal in raw_signals
        ]
        best_signal = self._choose_best_signal(raw_signals, scored_signals)
        enriched_content = self._compose_content(
            fallback_text, raw_signals, best_signal
        )
        summary = best_signal.content if best_signal else fallback_text[:2000]
        return MediaExtractionResult(
            media_type=media_type,
            summary=summary,
            signals=raw_signals,
            scores=scored_signals,
            best_signal=best_signal,
            enriched_content=enriched_content,
            metadata={
                "media_type": media_type.value,
                "signal_count": len(raw_signals),
                "best_signal": best_signal.source if best_signal else "fallback",
                "signal_scores": [asdict(score) for score in scored_signals],
            },
        )

    def _score_signal(
        self, signal: MediaSignal, fallback_text: str
    ) -> MediaSignalScore:
        content = signal.content.strip()
        word_count = max(len(content.split()), 1)
        coverage = (
            min(word_count / max(len(fallback_text.split()), 1), 1.0)
            if fallback_text
            else min(word_count / 1200, 1.0)
        )
        specificity = min(
            len(
                {
                    word.lower().strip(".,:;!?()[]{}")
                    for word in content.split()
                    if len(word) > 4
                }
            )
            / 40,
            1.0,
        )
        confidence = max(0.0, min(signal.confidence, 1.0))
        weighted_score = round(
            (coverage * 0.35) + (specificity * 0.35) + (confidence * 0.30), 4
        )
        return MediaSignalScore(
            signal_name=signal.name,
            coverage=round(coverage, 4),
            specificity=round(specificity, 4),
            confidence=round(confidence, 4),
            weighted_score=weighted_score,
        )

    def _choose_best_signal(
        self,
        signals: list[MediaSignal],
        scores: list[MediaSignalScore],
    ) -> MediaSignal | None:
        if not signals:
            return None
        ranked = sorted(
            zip(signals, scores, strict=False),
            key=lambda pair: pair[1].weighted_score,
            reverse=True,
        )
        return ranked[0][0]

    def _compose_content(
        self,
        fallback_text: str,
        signals: list[MediaSignal],
        best_signal: MediaSignal | None,
    ) -> str:
        parts: list[str] = []
        if fallback_text.strip():
            parts.append(fallback_text.strip())
        for signal in signals:
            if best_signal is not None and signal.source == best_signal.source:
                continue
            parts.append(f"[{signal.source}]\n{signal.content.strip()}")
        if best_signal is not None:
            parts.insert(
                0, f"[primary:{best_signal.source}]\n{best_signal.content.strip()}"
            )
        return "\n\n".join(parts).strip()
