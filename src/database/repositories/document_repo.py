import json
from collections.abc import Sequence

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import ChunkModel, DocumentModel
from src.shared.types import Chunk, Document


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_document(self, document: Document) -> DocumentModel:
        model = DocumentModel(
            id=document.id,
            title=document.title,
            source_path=document.source_path,
            source_type=document.source_type,
            author=document.author,
            department=document.department,
            tags=json.dumps(document.tags),
            metadata_json=json.dumps(document.metadata),
            s3_key=document.metadata.get("s3_key"),
            content_text=document.metadata.get("content_text", ""),
        )
        self.session.add(model)
        await self.session.flush()
        return model

    async def update_document_metadata(
        self,
        document_id: str,
        *,
        metadata_json: str | None = None,
        s3_key: str | None = None,
        content_text: str | None = None,
        bytes: int | None = None,
    ) -> int:
        values: dict[str, object] = {}
        if metadata_json is not None:
            values["metadata_json"] = metadata_json
        if s3_key is not None:
            values["s3_key"] = s3_key
        if content_text is not None:
            values["content_text"] = content_text
        if bytes is not None:
            values["bytes"] = bytes
        if not values:
            return 0
        result = await self.session.execute(
            update(DocumentModel)
            .where(DocumentModel.id == document_id)
            .values(**values)
        )
        return int(result.rowcount or 0)

    async def get_document(self, document_id: str) -> DocumentModel | None:
        return await self.session.get(DocumentModel, document_id)

    async def create_chunks(self, chunks: Sequence[Chunk]) -> list[ChunkModel]:
        models = [
            ChunkModel(
                id=c.id,
                document_id=c.document_id,
                content=c.content,
                parent_id=c.parent_id,
                page_number=c.page_number,
                position=c.position,
                section=c.section,
                entities=json.dumps(c.entities),
                metadata_json=json.dumps(c.metadata),
                vector_store_id=c.vector_store_id,
                attributes_json=json.dumps(c.attributes),
            )
            for c in chunks
        ]
        self.session.add_all(models)
        await self.session.flush()
        return models

    async def get_chunks_by_document(self, document_id: str) -> list[ChunkModel]:
        result = await self.session.execute(
            select(ChunkModel)
            .where(ChunkModel.document_id == document_id)
            .order_by(ChunkModel.position)
        )
        return list(result.scalars().all())

    async def get_chunk(self, chunk_id: str) -> ChunkModel | None:
        return await self.session.get(ChunkModel, chunk_id)

    async def get_documents_by_ids(
        self, document_ids: Sequence[str]
    ) -> dict[str, DocumentModel]:
        """Batch lookup documents by id. Returns a dict id -> DocumentModel."""
        if not document_ids:
            return {}
        unique_ids = list({i for i in document_ids if i})
        result = await self.session.execute(
            select(DocumentModel).where(DocumentModel.id.in_(unique_ids))
        )
        return {doc.id: doc for doc in result.scalars().all()}

    async def get_parent_chunks(self, document_id: str) -> list[ChunkModel]:
        result = await self.session.execute(
            select(ChunkModel).where(
                ChunkModel.document_id == document_id,
                ChunkModel.parent_id.is_(None),
            )
        )
        return list(result.scalars().all())

    async def get_document_by_id(self, document_id: str) -> DocumentModel | None:
        return await self.session.get(DocumentModel, document_id)

    async def list_documents(
        self,
        *,
        limit: int = 10000,
        after_id: str | None = None,
        order: str = "desc",
        purpose: str | None = None,
    ) -> list[DocumentModel]:
        stmt = select(DocumentModel)
        if purpose:
            stmt = stmt.where(
                DocumentModel.metadata_json.contains(f'"purpose": "{purpose}"')
            )
        if after_id:
            if order == "asc":
                stmt = stmt.where(DocumentModel.id > after_id)
            else:
                stmt = stmt.where(DocumentModel.id < after_id)
        stmt = stmt.order_by(
            DocumentModel.created_at.asc()
            if order == "asc"
            else DocumentModel.created_at.desc()
        )
        stmt = stmt.limit(limit + 1)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def delete_document(self, document_id: str) -> int:
        result = await self.session.execute(
            DocumentModel.__table__.delete().where(DocumentModel.id == document_id)
        )
        return int(result.rowcount or 0)

    async def update_chunks_vector_store_id(
        self, document_id: str, vector_store_id: str
    ) -> int:
        """Tag all existing chunks for a document with the given vector_store_id.

        Used when re-using already-indexed chunks across vector stores.
        Returns the number of rows updated.
        """
        result = await self.session.execute(
            update(ChunkModel)
            .where(ChunkModel.document_id == document_id)
            .values(vector_store_id=vector_store_id)
        )
        return int(result.rowcount or 0)

    async def get_chunks_by_ids(self, chunk_ids: Sequence[str]) -> list[ChunkModel]:
        if not chunk_ids:
            return []
        result = await self.session.execute(
            select(ChunkModel).where(ChunkModel.id.in_(list(chunk_ids)))
        )
        return list(result.scalars().all())

    async def get_chunks_by_vector_store_file(
        self, vector_store_id: str, document_id: str
    ) -> list[ChunkModel]:
        """Get chunks for a specific (vector_store_id, document_id) pair."""
        result = await self.session.execute(
            select(ChunkModel)
            .where(
                ChunkModel.vector_store_id == vector_store_id,
                ChunkModel.document_id == document_id,
            )
            .order_by(ChunkModel.position)
        )
        return list(result.scalars().all())

    async def delete_chunks_by_document(self, document_id: str) -> int:
        result = await self.session.execute(
            ChunkModel.__table__.delete().where(ChunkModel.document_id == document_id)
        )
        return int(result.rowcount or 0)
