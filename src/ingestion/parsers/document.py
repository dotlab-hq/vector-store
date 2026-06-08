from __future__ import annotations

import asyncio
import base64
from pathlib import Path

from langchain_core.messages import HumanMessage

from src.ingestion.parsers.base import MediaParser, ParserSignal
from src.llm.openai import llm


class DocumentParser(MediaParser):
    def can_parse(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in {
            ".pdf",
            ".docx",
            ".doc",
            ".pptx",
            ".ppt",
            ".odt",
        }

    async def parse(
        self, file_path: Path, file_bytes: bytes, fallback_text: str = ""
    ) -> ParserSignal | None:
        if file_path.suffix.lower() == ".pdf":
            page_signal = await self._parse_pdf_pages(file_path, fallback_text)
            if page_signal is not None:
                return page_signal
        try:
            from markitdown import MarkItDown

            md = MarkItDown()
            result = await asyncio.to_thread(md.convert_local, str(file_path))
            content = (result.text_content or "").strip()
            if not content:
                return None
            return ParserSignal(
                name="document_ocr",
                content=content,
                confidence=0.9,
                source="markitdown_document",
                metadata={
                    "page_strategy": "page_level_ocr_and_layout",
                    "fallback_text_length": len(fallback_text),
                },
            )
        except Exception:
            return None

    async def _parse_pdf_pages(
        self, file_path: Path, fallback_text: str
    ) -> ParserSignal | None:
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(file_path))
            page_texts = [(page.extract_text() or "").strip() for page in reader.pages]
            if any(page_texts):
                return await self._summarize_text_pages(page_texts, fallback_text)
            return await self._summarize_scanned_pages(
                file_path, len(reader.pages), fallback_text
            )
        except Exception:
            return None

    async def _summarize_text_pages(
        self, page_texts: list[str], fallback_text: str
    ) -> ParserSignal | None:
        async def summarize_page(page_number: int, page_text: str) -> str:
            if not page_text.strip():
                return ""
            response = await llm.ainvoke(
                [
                    HumanMessage(
                        content=(
                            "Extract the most useful information from this page. "
                            "Keep entities, numbers, headings, lists, and key facts.\n\n"
                            f"Page {page_number + 1}:\n{page_text}"
                        )
                    )
                ]
            )
            return (getattr(response, "content", "") or "").strip()

        summaries = await asyncio.gather(
            *(
                summarize_page(page_number, page_text)
                for page_number, page_text in enumerate(page_texts)
            )
        )
        merged = "\n\n".join(
            f"# Page {index + 1}\n{summary or page_text}"
            for index, (summary, page_text) in enumerate(
                zip(summaries, page_texts, strict=False)
            )
            if summary or page_text
        ).strip()
        if not merged:
            return None
        return ParserSignal(
            name="pdf_page_llm",
            content=merged,
            confidence=0.93,
            source="llm_pdf_pages",
            metadata={
                "page_count": len(page_texts),
                "page_strategy": "per_page_llm_summary",
                "fallback_text_length": len(fallback_text),
            },
        )

    async def _summarize_scanned_pages(
        self,
        file_path: Path,
        page_count: int,
        fallback_text: str,
    ) -> ParserSignal | None:
        try:
            import fitz  # PyMuPDF
        except Exception:
            return None

        try:
            doc = fitz.open(str(file_path))
        except Exception:
            return None

        async def describe_page(page_index: int) -> str:
            try:
                page = doc[page_index]
                pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                image_bytes = pixmap.tobytes("png")
                payload = base64.b64encode(image_bytes).decode("ascii")
                response = await llm.ainvoke(
                    [
                        HumanMessage(
                            content=[
                                {
                                    "type": "text",
                                    "text": (
                                        "Read this scanned PDF page like OCR and produce markdown. "
                                        "Preserve headings, lists, tables, numbers, and any visible text."
                                    ),
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{payload}"
                                    },
                                },
                            ]
                        )
                    ]
                )
                return (getattr(response, "content", "") or "").strip()
            except Exception:
                return ""

        descriptions = await asyncio.gather(
            *(describe_page(i) for i in range(len(doc)))
        )
        merged = "\n\n".join(
            f"# Page {index + 1}\n{description}".strip()
            for index, description in enumerate(descriptions)
            if description.strip()
        ).strip()
        if not merged:
            return None
        return ParserSignal(
            name="pdf_scanned_page_llm",
            content=merged,
            confidence=0.95,
            source="llm_pdf_scanned_pages",
            metadata={
                "page_count": page_count,
                "page_strategy": "scanned_page_multimodal",
                "fallback_text_length": len(fallback_text),
            },
        )
