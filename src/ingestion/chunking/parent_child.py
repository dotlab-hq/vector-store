from src.config import settings
from src.ingestion.chunking.semantic_chunker import SemanticChunker
from src.shared.types import Chunk


class ParentChildChunker:
    def __init__(self) -> None:
        self.parent_chunker = SemanticChunker(
            max_chunk_tokens=settings.parent_chunk_max_tokens
        )
        self.child_chunker = SemanticChunker(
            max_chunk_tokens=settings.child_chunk_max_tokens
        )

    def build(
        self,
        content: str,
        document_id: str,
    ) -> list[Chunk]:
        parent_semantics = self.parent_chunker.chunk(content)
        all_chunks: list[Chunk] = []
        chunk_counter = 0

        for parent_idx, parent in enumerate(parent_semantics):
            parent_id = f"{document_id}_p{parent_idx}"
            all_chunks.append(
                Chunk(
                    id=parent_id,
                    document_id=document_id,
                    content=parent.content,
                    page_number=parent.page_number,
                    position=chunk_counter,
                    section=parent.section,
                )
            )
            chunk_counter += 1

            child_semantics = self.child_chunker.chunk(parent.content)
            for child_idx, child in enumerate(child_semantics):
                all_chunks.append(
                    Chunk(
                        id=f"{parent_id}_c{child_idx}",
                        document_id=document_id,
                        content=child.content,
                        parent_id=parent_id,
                        # Inherit parent's page/section when child's is missing
                        page_number=child.page_number or parent.page_number,
                        position=chunk_counter,
                        section=child.section or parent.section,
                    )
                )
                chunk_counter += 1

        return all_chunks
