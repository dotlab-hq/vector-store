"""
Docling-based document loader — IBM's deep document parsing for PDF, DOCX, PPTX, HTML, CSV, and more.

Uses langchain-docling's DoclingLoader under the hood with MARKDOWN export,
then wraps results in our RawDocument format. Handles tables, formulas,
reading order, and scanned PDF OCR automatically.
"""

from __future__ import annotations

from pathlib import Path

import structlog

from src.ingestion.loaders.base import DocumentLoader, RawDocument

logger = structlog.get_logger()

DOCLING_EXTENSIONS = {
    # Documents
    ".pdf", ".docx", ".doc", ".odt", ".pptx", ".ppt", ".odp",
    # Spreadsheets
    ".xlsx", ".xls", ".ods", ".csv", ".tsv",
    # Web
    ".html", ".htm",
    # Text / Structured
    ".txt", ".md", ".json", ".xml", ".yaml", ".yml",
    # E-books
    ".epub",
}


class DoclingLoader(DocumentLoader):
    """
    Deep document parser powered by IBM Docling.

    Extracts text, tables, formulas, images metadata, and reading order from
    PDF/DOCX/PPTX/HTML. Handles both born-digital and scanned PDFs.

    Falls back to plain text for unsupported formats.
    """

    def __init__(self, *, use_gpu: bool = False) -> None:
        self._use_gpu = use_gpu
        self._initialized = False

    def _ensure_imports(self) -> None:
        if self._initialized:
            return
        try:
            import langchain_docling  # noqa: F401
            self._initialized = True
        except ImportError:
            logger.error("langchain_docling_not_installed")
            raise ImportError(
                "Install docling extras: "
                "`uv add langchain-docling docling`"
            )

    def can_load(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in DOCLING_EXTENSIONS

    def load(self, file_path: Path) -> RawDocument:
        self._ensure_imports()

        try:
            from langchain_docling import DoclingLoader
            from langchain_docling.loader import ExportType

            loader = DoclingLoader(
                file_path=str(file_path),
                export_type=ExportType.MARKDOWN,
            )
            documents = list(loader.load())

            if not documents:
                logger.warning("docling_empty_result", file=str(file_path))
                return self._fallback(file_path)

            # Docling produces one Document per page/section in MARKDOWN mode
            combined_content = "\n\n".join(
                doc.page_content for doc in documents if doc.page_content.strip()
            ).strip()

            if not combined_content:
                return self._fallback(file_path)

            # Merge metadata from all chunks
            metadata: dict = {
                "title": file_path.stem,
                "extension": file_path.suffix.lower(),
                "converter": "docling",
                "chunk_count": len(documents),
            }

            # Pull dl_meta from first document if present
            if documents and "dl_meta" in (documents[0].metadata or {}):
                metadata["dl_meta"] = documents[0].metadata["dl_meta"]
            if documents and "source" in (documents[0].metadata or {}):
                metadata["source_url"] = documents[0].metadata.get("source")

            return RawDocument(
                content=combined_content,
                metadata=metadata,
                source_path=str(file_path),
            )

        except ImportError:
            logger.warning("langchain_docling_missing", file=str(file_path))
            return self._fallback(file_path)
        except Exception as e:
            logger.warning(
                "docling_convert_failed",
                file=str(file_path),
                error=str(e),
            )
            return self._fallback(file_path)

    def _fallback(self, file_path: Path) -> RawDocument:
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            content = ""
        return RawDocument(
            content=content,
            metadata={"title": file_path.stem, "extension": file_path.suffix.lower(), "converter": "fallback_text"},
            source_path=str(file_path),
        )
