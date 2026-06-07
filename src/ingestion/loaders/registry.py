"""
Central loader registry — routes file extensions to the best available loader.

Priority order:
  1. Docling (IBM deep parsing — PDF/DOCX/PPTX/HTML/CSV)
  2. Format-specific loaders (pypdf, python-docx, python-pptx, bs4, openpyxl)
  3. MarkItDown (universal fallback for any other format)
"""

from __future__ import annotations

from pathlib import Path

import structlog

from src.ingestion.loaders.base import DocumentLoader

logger = structlog.get_logger()


class DocumentLoaderRegistry:
    """Select the best loader for a given file path based on format and availability."""

    def __init__(self) -> None:
        self._loaders: list[DocumentLoader] = []
        self._build_chain()

    def _build_chain(self) -> None:
        from src.ingestion.loaders.docling_loader import DoclingLoader
        from src.ingestion.loaders.format_loaders import (
            DOCXLoader,
            CSVLoader,
            ExcelLoader,
            HTMLLoader,
            JSONLoader,
            PDFLoader,
            PPTXLoader,
            TextLoader,
        )
        from src.ingestion.loaders.markitdown_loader import MarkItDownLoader

        # Format-specific loaders first (fastest, most targeted)
        self._loaders = [
            PDFLoader(),
            DOCXLoader(),
            PPTXLoader(),
            ExcelLoader(),
            CSVLoader(),
            JSONLoader(),
            HTMLLoader(),
            TextLoader(),
            # Docling — deep parsing fallback for unsupported or better-handled formats
            DoclingLoader(),
            # MarkItDown — universal fallback
            MarkItDownLoader(),
        ]

    def get_loader(self, file_path: Path) -> DocumentLoader:
        for loader in self._loaders:
            if loader.can_load(file_path):
                return loader
        # Should never reach here (MarkItDownLoader is a catch-all)
        raise ValueError(f"No loader found for file type: {file_path.suffix}")

    @property
    def supported_extensions(self) -> set[str]:
        exts: set[str] = set()
        for loader in self._loaders:
            if hasattr(loader, "DOCLING_EXTENSIONS"):
                exts |= loader.DOCLING_EXTENSIONS
            if hasattr(loader, "MARKITDOWN_EXTENSIONS"):
                exts |= loader.MARKITDOWN_EXTENSIONS
        # Manually cover format_loader extensions
        exts |= {
            ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls", ".ods",
            ".csv", ".tsv", ".json", ".html", ".htm", ".txt", ".md", ".markdown",
            ".rst", ".xml", ".yaml", ".yml",
        }
        return exts
