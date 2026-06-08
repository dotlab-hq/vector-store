from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from langchain_core.messages import HumanMessage

from src.ingestion.parsers.base import MediaParser, ParserSignal
from src.llm.openai import llm


class AudioParser(MediaParser):
    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in {".mp3", ".wav", ".m4a", ".ogg", ".flac"}

    async def parse(
        self, file_path: Path, file_bytes: bytes, fallback_text: str = ""
    ) -> ParserSignal | None:
        mime_type, _ = mimetypes.guess_type(str(file_path))
        try:
            payload = base64.b64encode(file_bytes).decode("ascii")
            response = await llm.ainvoke(
                [
                    HumanMessage(
                        content=[
                            {
                                "type": "text",
                                "text": (
                                    "Transcribe this audio and capture the maximum useful information, "
                                    "including speakers, entities, numbers, and actions."
                                ),
                            },
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": payload,
                                    "format": (mime_type or "audio/wav").split("/")[-1],
                                },
                            },
                        ]
                    )
                ]
            )
            content = getattr(response, "content", "") or ""
            if not content.strip():
                return None
            return ParserSignal(
                name="audio_transcript",
                content=content.strip(),
                confidence=0.9,
                source="llm_audio",
                metadata={
                    "mime_type": mime_type or "",
                    "fallback_text_length": len(fallback_text),
                },
            )
        except Exception:
            return None
