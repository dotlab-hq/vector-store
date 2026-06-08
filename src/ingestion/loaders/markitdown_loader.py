"""
Universal Document Loader — uses Microsoft MarkItDown for format-to-Markdown conversion.

Replaces individual format-specific loaders with a single, comprehensive converter.
MarkItDown handles: PDF, PowerPoint, Word, Excel, Images (EXIF + OCR),
Audio (EXIF + transcription), HTML, CSV, JSON, XML, ZIP, YouTube URLs, EPUB, and more.

Optional LLM integration for image descriptions and document analysis.
"""

from __future__ import annotations

import structlog
from pathlib import Path

from src.ingestion.loaders.base import DocumentLoader, RawDocument

logger = structlog.get_logger()

# All extensions MarkItDown supports
MARKITDOWN_EXTENSIONS = {
    # Documents
    ".pdf",
    ".docx",
    ".doc",
    ".odt",
    ".pptx",
    ".ppt",
    ".odp",
    # Spreadsheets
    ".xlsx",
    ".xls",
    ".ods",
    ".csv",
    ".tsv",
    # Text / Structured
    ".txt",
    ".md",
    ".json",
    ".xml",
    ".yaml",
    ".yml",
    ".html",
    ".htm",
    # Images
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
    ".svg",
    # Audio
    ".mp3",
    ".wav",
    ".m4a",
    ".ogg",
    ".flac",
    # Archives
    ".zip",
    # E-books
    ".epub",
    # Other
    ".url",
}


class MarkItDownLoader(DocumentLoader):
    """
    Universal document loader powered by Microsoft MarkItDown.

    Converts any supported file to Markdown while preserving structure:
    headings, lists, tables, links, code blocks, images metadata, etc.

    Supports optional LLM client for enhanced image descriptions.
    """

    def __init__(
        self,
        *,
        llm_client=None,
        llm_model: str = "gpt-4o",
    ) -> None:
        self._llm_client = llm_client
        self._llm_model = llm_model
        self._md = None

    def _get_converter(self):
        """Lazy-initialize the MarkItDown converter."""
        if self._md is not None:
            return self._md

        try:
            from markitdown import MarkItDown

            kwargs: dict = {}
            if self._llm_client is not None:
                kwargs["llm_client"] = self._llm_client
                kwargs["llm_model"] = self._llm_model

            self._md = MarkItDown(**kwargs)
            return self._md

        except ImportError:
            logger.error("markitdown_not_installed")
            raise

    def can_load(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in MARKITDOWN_EXTENSIONS

    def load(self, file_path: Path) -> RawDocument:
        md = self._get_converter()

        try:
            result = md.convert_local(str(file_path))
            content = result.text_content or ""
        except Exception as e:
            logger.warning(
                "markitdown_convert_failed",
                file=str(file_path),
                error=str(e),
            )
            # Fallback: read as plain text
            content = file_path.read_text(encoding="utf-8", errors="replace")

        metadata = {
            "title": file_path.stem,
            "extension": file_path.suffix.lower(),
            "converter": "markitdown",
        }

        return RawDocument(
            content=content,
            metadata=metadata,
            source_path=str(file_path),
        )
