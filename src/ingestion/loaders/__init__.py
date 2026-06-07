from .docling_loader import DoclingLoader
from .format_loaders import (
    CSVLoader,
    DOCXLoader,
    ExcelLoader,
    HTMLLoader,
    JSONLoader,
    PDFLoader,
    PPTXLoader,
    TextLoader,
    XMLLoader,
)
from .markitdown_loader import MarkItDownLoader
from .registry import DocumentLoaderRegistry

__all__ = [
    "CSVLoader",
    "DoclingLoader",
    "DOCXLoader",
    "DocumentLoaderRegistry",
    "ExcelLoader",
    "HTMLLoader",
    "JSONLoader",
    "MarkItDownLoader",
    "PDFLoader",
    "PPTXLoader",
    "TextLoader",
    "XMLLoader",
]
