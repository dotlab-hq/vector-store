from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from langchain_core.messages import HumanMessage

from src.ingestion.parsers.base import MediaParser, ParserSignal
from src.llm.openai import llm


class VideoParser(MediaParser):
    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi"}

    async def parse(self, file_path: Path, file_bytes: bytes, fallback_text: str = "") -> ParserSignal | None:
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
                                    "Summarize this video multimodally. Describe scenes, on-screen text, objects, "
                                    "actions, people, and any spoken content that can be inferred from the asset."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type or 'video/mp4'};base64,{payload}"},
                            },
                        ]
                    )
                ]
            )
            content = getattr(response, "content", "") or ""
            if not content.strip():
                return None
            return ParserSignal(
                name="video_multimodal",
                content=content.strip(),
                confidence=0.82,
                source="llm_video",
                metadata={"mime_type": mime_type or "", "fallback_text_length": len(fallback_text)},
            )
        except Exception:
            return None
