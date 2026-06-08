from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from langchain_core.messages import HumanMessage

from src.ingestion.parsers.base import MediaParser, ParserSignal
from src.llm.openai import llm


class ImageParser(MediaParser):
    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".bmp",
            ".tiff",
            ".gif",
        }

    async def parse(
        self, file_path: Path, file_bytes: bytes, fallback_text: str = ""
    ) -> ParserSignal | None:
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if not mime_type:
            return None
        try:
            payload = base64.b64encode(file_bytes).decode("ascii")
            response = await llm.ainvoke(
                [
                    HumanMessage(
                        content=[
                            {
                                "type": "text",
                                "text": (
                                    "Describe this image with maximum information density. "
                                    "Extract OCR text, objects, people, layout, charts, and visible context."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{payload}"
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
                name="image_multimodal",
                content=content.strip(),
                confidence=0.94,
                source="llm_image",
                metadata={
                    "mime_type": mime_type,
                    "fallback_text_length": len(fallback_text),
                },
            )
        except Exception:
            return None
