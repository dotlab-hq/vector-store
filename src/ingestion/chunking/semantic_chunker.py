import re

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from pydantic import BaseModel

from src.config import settings


class SemanticChunk(BaseModel):
    content: str
    section: str = ""
    page_number: int | None = None


class SemanticChunker:
    def __init__(
        self,
        max_chunk_tokens: int = settings.parent_chunk_max_tokens,
    ) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_tokens * 4,  # rough token-to-char ratio
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " "],
        )
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "heading_1"),
                ("##", "heading_2"),
                ("###", "heading_3"),
            ],
        )

    def chunk(self, content: str) -> list[SemanticChunk]:
        current_section = ""
        page_number = None
        chunks: list[SemanticChunk] = []

        sections = self.splitter.split_text(content)

        for section_text in sections:
            page_match = re.match(r"\[PAGE (\d+)\]", section_text)
            if page_match:
                page_number = int(page_match.group(1))
                section_text = section_text[page_match.end() :].strip()

            if not section_text:
                continue

            # Detect section headers
            header_match = re.match(r"^(#{1,3})\s+(.+)", section_text)
            if header_match:
                current_section = header_match.group(2)

            chunks.append(
                SemanticChunk(
                    content=section_text,
                    section=current_section,
                    page_number=page_number,
                )
            )

        return chunks
