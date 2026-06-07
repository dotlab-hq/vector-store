import json
from collections.abc import Sequence

from langchain_openai import OpenAIEmbeddings
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.repositories import DocumentRepository
from src.indexing.bm25.bm25_store import Bm25Store
from src.indexing.qdrant.qdrant_store import QdrantVectorStore
from src.observability.logging import get_logger
from src.shared.types import Chunk

logger = get_logger()


class Indexer:
    def __init__(
        self,
        session: AsyncSession,
        embedding_provider: OpenAIEmbeddings,
        qdrant_store: QdrantVectorStore,
        bm25_store: Bm25Store,
    ) -> None:
        self.session = session
        self.repo = DocumentRepository(session)
        self.embedder = embedding_provider
        self.qdrant = qdrant_store
        self.bm25 = bm25_store

    async def index_document(self, document_id: str, batch_size: int = 15) -> None:
        chunks_models = await self.repo.get_chunks_by_document(document_id)
        if not chunks_models:
            logger.warning("no_chunks_found", document_id=document_id)
            return

        chunks = [
            Chunk(
                id=cm.id,
                document_id=cm.document_id,
                content=cm.content,
                parent_id=cm.parent_id,
                page_number=cm.page_number,
                position=cm.position,
                section=cm.section,
                vector_store_id=cm.vector_store_id,
                attributes=json.loads(cm.attributes_json or "{}"),
            )
            for cm in chunks_models
        ]

        total_batches = (len(chunks) + batch_size - 1) // batch_size

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            texts = [c.content for c in batch]

            try:
                embeddings = await self.embedder.aembed_documents(texts)
                if embeddings:
                    logger.info(
                        "embedding_dimension",
                        document_id=document_id,
                        dimension=len(embeddings[0]),
                        batch_size=len(embeddings),
                    )
                await self.qdrant.insert(batch, embeddings)
                await self.bm25.insert(batch)
                logger.info(
                    "batch_indexed",
                    document_id=document_id,
                    batch=f"{batch_num}/{total_batches}",
                    batch_size=len(batch),
                )
            except Exception as e:
                import traceback
                logger.error(
                    "batch_indexing_failed",
                    document_id=document_id,
                    batch=f"{batch_num}/{total_batches}",
                    error=str(e),
                    error_type=type(e).__name__,
                    traceback=traceback.format_exc(),
                )
                raise

        logger.info(
            "document_indexed",
            document_id=document_id,
            total_chunks=len(chunks),
        )

    async def search_hybrid(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 20,
    ) -> Sequence:
        dense_results = await self.qdrant.search(query_embedding, top_k=top_k)
        sparse_results = await self.bm25.search(query, top_k=top_k)
        return list(dense_results) + list(sparse_results)
