"""PDF diagram and visual element extractor.

Extracts diagrams, charts, flowcharts, and other visual elements from PDF pages
by rendering pages to images and analyzing them with the vision LLM. Each extracted
diagram becomes a separate chunk with rich metadata for retrieval.
"""

from __future__ import annotations

import asyncio
import base64
import mimetypes
from pathlib import Path

from langchain_core.messages import HumanMessage

from src.config import settings
from src.llm import vision_llm
from src.observability.logging import get_logger

logger = get_logger()


class PDFDiagramExtractor:
    """Extracts and analyzes diagrams/charts from PDF pages using vision LLM.

    Workflow:
    1. Render PDF page to image using PyMuPDF
    2. Detect visual regions (diagrams, charts, tables-as-images)
    3. Analyze each region with vision LLM for rich text description
    4. Return structured results for chunking/indexing
    """

    def __init__(self) -> None:
        self.render_dpi = settings.pdf_diagram_render_dpi
        self.render_matrix = settings.pdf_diagram_render_matrix
        self.min_region_ratio = settings.pdf_diagram_min_region_ratio
        self.max_regions = settings.pdf_diagram_max_regions_per_page
        self.image_min_bytes = settings.pdf_diagram_image_min_bytes
        self.diagram_prompt = settings.pdf_diagram_prompt
        self.vision_page_prompt = settings.pdf_vision_page_prompt

    async def extract_page_as_image(
        self, file_path: Path, page_index: int
    ) -> bytes | None:
        """Render a single PDF page to PNG bytes."""
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(str(file_path))
            if page_index >= len(doc):
                return None
            page = doc[page_index]
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(self.render_matrix, self.render_matrix),
                alpha=False,
            )
            image_bytes = pixmap.tobytes("png")
            doc.close()
            return image_bytes
        except ImportError:
            logger.warning("pymupdf_not_installed")
            return None
        except Exception as exc:
            logger.warning(
                "pdf_page_render_failed",
                page=page_index,
                error=str(exc),
            )
            return None

    async def extract_page_as_images(
        self, file_path: Path
    ) -> list[tuple[int, bytes]]:
        """Render all pages of a PDF to PNG bytes. Returns (page_index, image_bytes)."""
        try:
            import fitz

            doc = fitz.open(str(file_path))
            total_pages = len(doc)
            results: list[tuple[int, bytes]] = []

            async def render_page(idx: int) -> tuple[int, bytes] | None:
                try:
                    page = doc[idx]
                    pixmap = page.get_pixmap(
                        matrix=fitz.Matrix(self.render_matrix, self.render_matrix),
                        alpha=False,
                    )
                    return (idx, pixmap.tobytes("png"))
                except Exception:
                    return None

            tasks = [render_page(i) for i in range(total_pages)]
            rendered = await asyncio.gather(*tasks)
            for r in rendered:
                if r is not None:
                    results.append(r)
            doc.close()
            return results
        except Exception as exc:
            logger.warning("pdf_pages_render_failed", error=str(exc))
            return []

    async def _analyze_page_image(
        self, image_bytes: bytes, page_number: int, mime_type: str = "image/png"
    ) -> str | None:
        """Send a page image to the vision LLM for analysis."""
        payload = base64.b64encode(image_bytes).decode("ascii")
        try:
            response = await vision_llm.ainvoke(
                [
                    HumanMessage(
                        content=[
                            {
                                "type": "text",
                                "text": self.vision_page_prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{payload}",
                                    "detail": "high",
                                },
                            },
                        ]
                    )
                ]
            )
            content = getattr(response, "content", "") or ""
            if not content.strip():
                return None

            meta = getattr(response, "response_metadata", None) or {}
            usage = meta.get("token_usage") or meta.get("usage") or {}
            tokens_in = int(usage.get("prompt_tokens", 0) or 0)
            tokens_out = int(usage.get("completion_tokens", 0) or 0)

            logger.info(
                "vision_page_analyzed",
                page=page_number,
                content_length=len(content),
                tokens_input=tokens_in,
                tokens_output=tokens_out,
            )
            return content.strip()
        except Exception as exc:
            logger.warning(
                "vision_page_analysis_failed",
                page=page_number,
                error=str(exc),
            )
            return None

    async def _analyze_diagram_region(
        self, image_bytes: bytes, region_label: str, page_number: int
    ) -> str | None:
        """Analyze a specific diagram region with the vision LLM."""
        payload = base64.b64encode(image_bytes).decode("ascii")
        try:
            prompt = (
                f"{self.diagram_prompt}\n\n"
                f"This is extracted from page {page_number} of a PDF document. "
                f"Region context: {region_label}"
            )
            response = await vision_llm.ainvoke(
                [
                    HumanMessage(
                        content=[
                            {
                                "type": "text",
                                "text": prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{payload}",
                                    "detail": "high",
                                },
                            },
                        ]
                    )
                ]
            )
            content = getattr(response, "content", "") or ""
            if not content.strip():
                return None

            meta = getattr(response, "response_metadata", None) or {}
            usage = meta.get("token_usage") or meta.get("usage") or {}
            tokens_in = int(usage.get("prompt_tokens", 0) or 0)
            tokens_out = int(usage.get("completion_tokens", 0) or 0)

            logger.info(
                "vision_diagram_analyzed",
                page=page_number,
                region=region_label,
                content_length=len(content),
                tokens_input=tokens_in,
                tokens_output=tokens_out,
            )
            return content.strip()
        except Exception as exc:
            logger.warning(
                "vision_diagram_analysis_failed",
                page=page_number,
                region=region_label,
                error=str(exc),
            )
            return None

    async def extract_page_diagrams(
        self, file_path: Path, page_index: int
    ) -> list[dict]:
        """Extract individual diagram images from a rendered PDF page.

        Uses PyMuPDF image extraction to find embedded images within the page,
        then analyzes each with the vision LLM.
        """
        try:
            import fitz

            doc = fitz.open(str(file_path))
            if page_index >= len(doc):
                doc.close()
                return []

            page = doc[page_index]
            image_list = page.get_images(full=True)
            diagrams: list[dict] = []

            for img_idx, img_info in enumerate(image_list[: self.max_regions]):
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    if len(image_bytes) < self.image_min_bytes:
                        continue

                    img_ext = base_image.get("ext", "png")
                    mime_type = mimetypes.guess_type(f"dummy.{img_ext}")[0]
                    if not mime_type:
                        mime_type = "image/png"

                    description = await self._analyze_diagram_region(
                        image_bytes,
                        region_label=f"embedded image {img_idx + 1}",
                        page_number=page_index + 1,
                    )
                    if description:
                        diagrams.append(
                            {
                                "page_index": page_index,
                                "page_number": page_index + 1,
                                "region_index": img_idx,
                                "description": description,
                                "image_bytes": image_bytes,
                                "mime_type": mime_type,
                                "source": "embedded_image",
                            }
                        )
                except Exception as exc:
                    logger.warning(
                        "embedded_image_extraction_failed",
                        page=page_index,
                        img_idx=img_idx,
                        error=str(exc),
                    )

            doc.close()
            return diagrams
        except ImportError:
            return []
        except Exception as exc:
            logger.warning(
                "diagram_extraction_failed",
                page=page_index,
                error=str(exc),
            )
            return []

    async def extract_all_pages(
        self, file_path: Path
    ) -> list[dict]:
        """Render every page and analyze with vision LLM.

        Returns one dict per page with keys:
        - page_number: 1-based page index
        - description: vision LLM analysis of the page content
        - diagrams: list of extracted embedded diagram analyses
        """
        try:
            import fitz
        except ImportError:
            logger.warning("pymupdf_not_installed_skipping_vision_extraction")
            return []

        doc = fitz.open(str(file_path))
        total_pages = len(doc)
        doc.close()

        results: list[dict] = []

        async def process_page(page_idx: int) -> dict | None:
            page_number = page_idx + 1
            page_image = await self.extract_page_as_image(file_path, page_idx)
            if page_image is None:
                return None

            # Analyze full page
            description = await self._analyze_page_image(
                page_image, page_number
            )

            # Extract embedded diagrams
            diagrams = await self.extract_page_diagrams(file_path, page_idx)

            return {
                "page_number": page_number,
                "page_image": page_image,
                "description": description or "",
                "diagrams": diagrams,
            }

        # Process pages concurrently (respect vision LLM rate limits)
        concurrency = min(8, settings.vision_max_tokens // 512)
        sem = asyncio.Semaphore(max(concurrency, 1))

        async def bounded_process(idx: int) -> dict | None:
            async with sem:
                return await process_page(idx)

        tasks = [bounded_process(i) for i in range(total_pages)]
        results_raw = await asyncio.gather(*tasks)

        for r in results_raw:
            if r is not None:
                results.append(r)

        logger.info(
            "pdf_vision_extraction_done",
            file=str(file_path),
            total_pages=total_pages,
            pages_with_description=sum(1 for r in results if r["description"]),
            total_diagrams=sum(len(r["diagrams"]) for r in results),
        )

        return results
