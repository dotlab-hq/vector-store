import json
from dataclasses import asdict
from pathlib import Path
import tempfile
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.database.repositories import DocumentRepository
from src.ingestion.chunking.parent_child import ParentChildChunker
from src.ingestion.loaders.base import DocumentLoader
from src.ingestion.loaders.registry import DocumentLoaderRegistry
from src.ingestion.media.processor import MediaProcessingService
from src.ingestion.metadata.extractor import MetadataExtractor
from src.indexing.indexer import Indexer
from src.observability.logging import get_logger
from src.shared.types import Chunk, Document

logger = get_logger()


class IngestionPipeline:
    def __init__(
        self,
        session: AsyncSession,
        embedder=None,
        qdrant_store=None,
        bm25_store=None,
    ) -> None:
        self.session = session
        self.repo = DocumentRepository(session)
        self.chunker = ParentChildChunker()
        self.extractor = MetadataExtractor()
        self._loader_registry = DocumentLoaderRegistry()
        self.media_service = MediaProcessingService()
        self._embedder = embedder
        self._qdrant_store = qdrant_store
        self._bm25_store = bm25_store

    def _get_loader(self, file_path: Path) -> DocumentLoader:
        return self._loader_registry.get_loader(file_path)

    async def _index_chunks(self, document_id: str) -> None:
        """Index document chunks into Qdrant and BM25 after DB storage."""
        if not all([self._qdrant_store, self._bm25_store, self._embedder]):
            logger.warning("stores_not_initialized", document_id=document_id)
            return

        try:
            indexer = Indexer(
                self.session, self._embedder, self._qdrant_store, self._bm25_store
            )
            await indexer.index_document(document_id)
        except Exception as e:
            import traceback

            logger.warning(
                "indexing_failed",
                document_id=document_id,
                error=str(e),
                error_type=type(e).__name__,
                traceback=traceback.format_exc(),
            )

    async def _store_in_s3(self, file_path: Path, document_id: str) -> str | None:
        """Store original file in S3. Returns the S3 key or None if S3 is not configured."""
        if not settings.s3_access_key:
            return None
        try:
            from src.storage.s3.client import S3Client

            s3 = S3Client()
            key = f"raw/{document_id}/{file_path.name}"
            data = file_path.read_bytes()
            await s3.upload(key, data)
            return key
        except Exception as e:
            logger.warning("s3_upload_failed", document_id=document_id, error=str(e))
            return None

    async def ingest(self, file_path: Path, title_override: str = "") -> Document:
        loader = self._get_loader(file_path)
        raw = loader.load(file_path)
        media_result = await self.media_service.process(
            file_path, fallback_text=raw.content
        )
        raw.content = media_result.enriched_content or raw.content
        raw.metadata.update(media_result.metadata)
        raw.metadata["media_signals"] = [
            {
                "name": signal.name,
                "source": signal.source,
                "confidence": signal.confidence,
                "content": signal.content,
                "metadata": signal.metadata,
            }
            for signal in media_result.signals
        ]
        raw.metadata["media_scores"] = [asdict(score) for score in media_result.scores]
        raw.metadata["media_summary"] = media_result.summary

        if not raw.content.strip():
            raw.content = raw.metadata.get("media_summary", "") or raw.metadata.get(
                "title", file_path.stem
            )
            raw.metadata["content_fallback"] = "summary_only"

        document = self.extractor.extract(raw, file_path)
        if not document.id:
            document.id = uuid4().hex
        if title_override:
            document.title = title_override
        if raw.metadata.get("media_type"):
            document.metadata["media_type"] = raw.metadata["media_type"]
        if raw.metadata.get("media_scores"):
            document.metadata["media_scores"] = raw.metadata["media_scores"]
        if raw.metadata.get("media_summary"):
            document.metadata["media_summary"] = raw.metadata["media_summary"]

        # Store original in S3
        s3_key = await self._store_in_s3(file_path, document.id)
        if s3_key:
            document.metadata["s3_key"] = s3_key

        await self.repo.create_document(document)
        logger.info("document_created", document_id=document.id, title=document.title)

        chunks = self.chunker.build(raw.content, document.id)
        await self.repo.create_chunks(chunks)
        logger.info("chunks_created", document_id=document.id, count=len(chunks))

        await self.session.commit()

        # Index chunks into Qdrant + BM25 (so they're searchable)
        await self._index_chunks(document.id)

        # Ingest into Knowledge Graph if enabled
        await self._ingest_to_kg(document.id, chunks)

        return document

    async def _set_processing_status(self, document_id: str, status: str) -> None:
        """Update the processing_status in document metadata."""
        try:
            row = await self.repo.get_document(document_id)
            if row is None:
                return
            meta = json.loads(row.metadata_json or "{}")
            meta["processing_status"] = status
            await self.repo.update_document_metadata(
                document_id, metadata_json=json.dumps(meta)
            )
            await self.session.commit()
        except Exception:
            pass

    async def process_existing_document(self, document_id: str) -> Document | None:
        """Process a stored document asynchronously after upload."""
        document = await self.repo.get_document(document_id)
        if document is None:
            logger.warning("document_not_found_for_processing", document_id=document_id)
            return None
        if not document.s3_key:
            logger.warning("document_missing_s3_key", document_id=document_id)
            await self._set_processing_status(document_id, "error")
            return None

        try:
            existing_metadata = json.loads(document.metadata_json or "{}")
        except Exception:
            existing_metadata = {}

        try:
            from src.storage.s3.client import S3Client

            s3 = S3Client()
            file_bytes = await s3.download(document.s3_key)
        except Exception as exc:
            logger.warning(
                "document_download_failed",
                document_id=document_id,
                s3_key=document.s3_key,
                error=str(exc),
            )
            await self._set_processing_status(document_id, "error")
            return None

        suffix = (
            Path(document.source_path or document.title or "upload.bin").suffix
            or ".bin"
        )
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = Path(tmp.name)

        try:
            loader = self._get_loader(tmp_path)
            raw = loader.load(tmp_path)
            media_result = await self.media_service.process(
                tmp_path, fallback_text=raw.content
            )
            raw.content = media_result.enriched_content or raw.content
            raw.metadata.update(media_result.metadata)
            raw.metadata["media_signals"] = [
                {
                    "name": signal.name,
                    "source": signal.source,
                    "confidence": signal.confidence,
                    "content": signal.content,
                    "metadata": signal.metadata,
                }
                for signal in media_result.signals
            ]
            raw.metadata["media_scores"] = [
                asdict(score) for score in media_result.scores
            ]
            raw.metadata["media_summary"] = media_result.summary

            if not raw.content.strip():
                raw.content = raw.metadata.get("media_summary", "") or raw.metadata.get(
                    "title", document.title
                )
                raw.metadata["content_fallback"] = "summary_only"

            updated_metadata = {
                **existing_metadata,
                **raw.metadata,
                "processing_status": "processing",
            }
            updated_title = document.title or raw.metadata.get("title", document.title)

            await self.repo.update_document_metadata(
                document_id,
                metadata_json=json.dumps(updated_metadata),
                content_text=raw.content,
            )
            await self.repo.delete_chunks_by_document(document_id)
            chunks = self.chunker.build(raw.content, document.id)
            await self.repo.create_chunks(chunks)
            await self.session.commit()

            await self._index_chunks(document.id)
            await self._ingest_to_kg(document.id, chunks)
            updated_metadata["processing_status"] = "processed"
            await self.repo.update_document_metadata(
                document_id,
                metadata_json=json.dumps(updated_metadata),
                content_text=raw.content,
            )
            await self.session.commit()
            logger.info(
                "document_processed_async",
                document_id=document.id,
                chunk_count=len(chunks),
            )
            return Document(
                id=document.id,
                title=updated_title,
                source_path=document.source_path,
                source_type=document.source_type,
                author=document.author or "",
                department=document.department or "",
                tags=[],
                metadata=updated_metadata,
                content_text=raw.content,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

    async def ingest_text(self, text: str, title: str = "Untitled") -> Document:
        document = Document(
            id=uuid4().hex,
            title=title,
            source_path="",
            source_type="text",
            content_text=text,
        )

        await self.repo.create_document(document)
        chunks = self.chunker.build(text, document.id)
        await self.repo.create_chunks(chunks)
        await self.session.commit()

        # Index chunks into FAISS + BM25
        await self._index_chunks(document.id)

        # Ingest into Knowledge Graph if enabled
        await self._ingest_to_kg(document.id, chunks)

        return document

    async def _ingest_to_kg(self, document_id: str, chunks: list[Chunk]) -> None:
        """Ingest chunks into Neo4j knowledge graph if enabled."""
        if not settings.neo4j_enabled:
            return
        try:
            from src.graph.knowledge.graph_store import KnowledgeGraphStore
            from src.graph.knowledge.neo4j_client import Neo4jClient

            neo4j = Neo4jClient()
            kg = KnowledgeGraphStore(neo4j)
            chunk_dicts = [{"id": c.id, "content": c.content} for c in chunks]
            await kg.ingest_document(document_id, chunk_dicts)
            await neo4j.close()
        except Exception as e:
            logger.warning("kg_ingestion_failed", document_id=document_id, error=str(e))
