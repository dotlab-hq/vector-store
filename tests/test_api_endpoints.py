from __future__ import annotations

import json
import unittest
from contextlib import ExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from apps.api.schemas.vector_stores import (
    DeleteResponse,
    FileContentItem,
    FileContentResponse,
    FileCounts,
    ListVectorStoreFilesResponse,
    ListVectorStoresResponse,
    SearchResponse,
    SearchResultItem,
    ContentBlock,
    VectorStoreFileBatchFileCounts,
    VectorStoreFileBatchObject,
    VectorStoreFileObject,
    VectorStoreObject,
)
from src.shared.types import QueryIntent


def _now():
    from datetime import datetime

    return datetime.utcnow()


class _FakeEngineBegin:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def run_sync(self, fn):
        return None


class _FakeEngine:
    def begin(self):
        return _FakeEngineBegin()


class _FakeScheduler:
    def start(self):
        return None

    async def stop(self):
        return None


class _FakeWorkflow:
    async def ainvoke(self, state):
        return {
            "original_query": state.original_query,
            "response": "Test answer",
            "short_answer": "Short",
            "reranked_results": [],
            "retrieved_chunks": [],
            "reranked_chunks": [],
            "supporting_chunks": [],
            "intent": QueryIntent.SIMPLE,
            "context": "",
            "faithfulness_score": 1.0,
            "answer_relevance_score": 1.0,
            "context_recall_score": 1.0,
            "confidence": 1.0,
            "faithfulness_passed": True,
            "claim_count": 0,
            "supported_claims": 0,
            "tokens_input": 0,
            "tokens_output": 0,
        }


@dataclass
class _DocRow:
    id: str
    title: str
    created_at: object
    bytes: int = 0
    metadata_json: str = "{}"
    s3_key: str | None = None
    content_text: str = ""


@dataclass
class _ChunkRow:
    id: str
    document_id: str
    content: str
    parent_id: str | None = None
    page_number: int | None = None
    section: str = ""
    position: int = 0


@dataclass
class _VSRow:
    id: str
    name: str = "store"
    status: str = "in_progress"
    created_at: object = None
    last_active_at: object = None
    expires_at: object = None
    expires_after_days: int | None = None
    metadata_json: str = "{}"
    chunking_strategy: str = "auto"
    chunk_size_tokens: int | None = None
    chunk_overlap_tokens: int | None = None
    file_counts_json: str = "{}"
    usage_bytes: int = 0


@dataclass
class _VFRow:
    id: str
    vector_store_id: str
    source_document_id: str
    status: str = "pending"
    created_at: object = None
    completed_at: object | None = None
    failure_reason: str | None = None
    attempts: int = 0
    next_attempt_at: object | None = None
    locked_at: object | None = None
    locked_by: str | None = None
    bytes: int = 0
    attributes_json: str = "{}"
    batch_id: str | None = None


@dataclass
class _VFBRow:
    id: str
    vector_store_id: str
    status: str = "in_progress"
    created_at: object = None
    cancelled_at: object | None = None
    completed_at: object | None = None
    file_counts_json: str = "{}"
    attributes_json: str = "{}"


class _FakeDocumentRepository:
    def __init__(self, session):
        self.session = session

    async def list_documents(self, **kwargs):
        return [
            _DocRow(
                id="file-1",
                title="alpha.pdf",
                created_at=_now(),
                bytes=123,
                metadata_json=json.dumps(
                    {
                        "purpose": "assistants",
                        "processing_status": "processing",
                    }
                ),
            )
        ]

    async def get_document_by_id(self, document_id: str):
        if document_id == "missing":
            return None
        return _DocRow(
            id=document_id,
            title="alpha.pdf",
            created_at=_now(),
            bytes=123,
            metadata_json=json.dumps({"purpose": "assistants", "processing_status": "uploaded"}),
            s3_key="files/alpha.pdf",
            content_text="hello world",
        )

    async def create_document(self, document):
        return _DocRow(
            id=document.id,
            title=document.title,
            created_at=_now(),
            bytes=123,
            metadata_json=json.dumps(document.metadata),
            s3_key=document.metadata.get("s3_key"),
            content_text=document.metadata.get("content_text", ""),
        )

    async def update_document_metadata(self, *args, **kwargs):
        return 1

    async def delete_document(self, document_id: str):
        return 1

    async def delete_chunks_by_document(self, document_id: str):
        return 1

    async def get_chunks_by_document(self, document_id: str):
        return [_ChunkRow(id="chunk-1", document_id=document_id, content="chunk body", position=0)]

    async def get_chunks_by_ids(self, chunk_ids):
        return [_ChunkRow(id=cid, document_id="doc-1", content="chunk text", position=0) for cid in chunk_ids]

    async def get_chunks_by_vector_store_file(self, vector_store_id: str, document_id: str):
        return [_ChunkRow(id="chunk-1", document_id=document_id, content="vector store content", position=0)]

    async def get_documents_by_ids(self, document_ids):
        return {
            doc_id: _DocRow(
                id=doc_id,
                title=f"{doc_id}.pdf",
                created_at=_now(),
                s3_key=f"files/{doc_id}.pdf",
            )
            for doc_id in document_ids
        }


class _FakeS3:
    def __init__(self):
        self.uploaded: dict[str, bytes] = {}

    async def upload(self, key: str, data: bytes, content_type: str | None = None):
        self.uploaded[key] = data

    async def download(self, key: str) -> bytes:
        return self.uploaded.get(key, b"hello world")

    async def delete(self, key: str):
        self.uploaded.pop(key, None)


class _FakeIngestionPipeline:
    def __init__(self, session):
        self.session = session

    async def ingest_text(self, text: str, title: str = "Untitled"):
        return SimpleNamespace(id="doc-ingest-text", title=title)

    async def ingest(self, path: Path, title_override: str = "Untitled"):
        return SimpleNamespace(id="doc-ingest-file", title=title_override)

    async def process_existing_document(self, document_id: str):
        return None


class _FakeVectorStoreService:
    def __init__(self):
        now = _now()
        self.store = VectorStoreObject(
            id="vs-1",
            created_at=int(now.timestamp()),
            name="store",
            bytes=0,
            status="in_progress",
            file_counts=FileCounts(),
            last_active_at=int(now.timestamp()),
            metadata={},
            expires_after=None,
            expires_at=None,
        )
        self.file = VectorStoreFileObject(
            id="file-1",
            created_at=int(now.timestamp()),
            vector_store_id="vs-1",
            status="in_progress",
            bytes=0,
            usage_bytes=0,
            chunking_strategy=None,
            attributes={},
        )
        self.batch = VectorStoreFileBatchObject(
            id="vsfb-1",
            created_at=int(now.timestamp()),
            vector_store_id="vs-1",
            status="in_progress",
            file_counts=VectorStoreFileBatchFileCounts(),
        )

    async def create_store(self, request):
        return self.store

    async def list_stores(self, **kwargs):
        return ListVectorStoresResponse(data=[self.store], has_more=False, first_id=self.store.id, last_id=self.store.id)

    async def get_store(self, store_id: str):
        return self.store if store_id == self.store.id else None

    async def update_store(self, store_id: str, request):
        return self.store if store_id == self.store.id else None

    async def delete_store(self, store_id: str):
        return DeleteResponse(id=store_id, object="vector_store.deleted", deleted=True) if store_id == self.store.id else None

    async def attach_file(self, store_id: str, request):
        return self.file if store_id == self.store.id else None

    async def list_files(self, store_id: str, **kwargs):
        if store_id != self.store.id:
            return ListVectorStoreFilesResponse(object="list", data=[], has_more=False, first_id=None, last_id=None)
        return ListVectorStoreFilesResponse(object="list", data=[self.file], has_more=False, first_id=self.file.id, last_id=self.file.id)

    async def get_file(self, store_id: str, file_id: str):
        return self.file if store_id == self.store.id and file_id == self.file.id else None

    async def update_file_attributes(self, store_id: str, file_id: str, request):
        return self.file if store_id == self.store.id and file_id == self.file.id else None

    async def delete_file(self, store_id: str, file_id: str):
        return DeleteResponse(id=file_id, object="vector_store.file.deleted", deleted=True) if store_id == self.store.id and file_id == self.file.id else None

    async def get_file_content(self, store_id: str, file_id: str):
        return FileContentResponse(
            data=[FileContentItem(type="text", text="chunk body")],
            has_more=False,
            next_page=None,
            file_id=file_id,
            filename="alpha.pdf",
            attributes={},
        ) if store_id == self.store.id and file_id == self.file.id else None

    async def create_batch(self, store_id: str, request):
        return self.batch if store_id == self.store.id else None

    async def get_batch(self, store_id: str, batch_id: str):
        return self.batch if store_id == self.store.id and batch_id == self.batch.id else None

    async def list_batch_files(self, store_id: str, batch_id: str, **kwargs):
        if store_id != self.store.id or batch_id != self.batch.id:
            return None
        return ListVectorStoreFilesResponse(object="list", data=[self.file], has_more=False, first_id=self.file.id, last_id=self.file.id)

    async def cancel_batch(self, store_id: str, batch_id: str):
        return self.batch if store_id == self.store.id and batch_id == self.batch.id else None

    async def search(self, store_id: str, request):
        if store_id != self.store.id:
            return None
        return SearchResponse(
            data=[
                SearchResultItem(
                    file_id="file-1",
                    filename="alpha.pdf",
                    score=0.99,
                    attributes={},
                    content=[ContentBlock(type="text", text="chunk body")],
                )
            ],
            has_more=False,
            next_page=None,
            search_query=[request.query] if isinstance(request.query, str) else request.query,
        )


@asynccontextmanager
async def _fake_session_factory():
    class _Session:
        async def commit(self):
            return None

        async def rollback(self):
            return None

    yield _Session()


class ApiEndpointTests(unittest.TestCase):
    def setUp(self):
        import apps.api.main as api_main
        import apps.api.routes.documents as documents_route
        import apps.api.routes.files as files_route
        import apps.api.routes.ingestion as ingestion_route
        import apps.api.routes.query as query_route
        import apps.api.routes.vector_stores as vs_route

        self.stack = ExitStack()
        self.addCleanup(self.stack.close)

        fake_engine = _FakeEngine()
        fake_scheduler = _FakeScheduler()
        fake_workflow = _FakeWorkflow()
        fake_service = _FakeVectorStoreService()

        self.stack.enter_context(patch.object(api_main, "engine", fake_engine))
        self.stack.enter_context(patch.object(api_main, "init_dependencies", lambda: None))
        self.stack.enter_context(patch.object(api_main, "init_vector_store_scheduler", lambda: None))
        self.stack.enter_context(patch.object(api_main, "get_workflow", lambda: fake_workflow))
        self.stack.enter_context(patch.object(api_main, "get_scheduler", lambda: fake_scheduler))

        self.stack.enter_context(patch.object(files_route, "DocumentRepository", _FakeDocumentRepository))
        self.stack.enter_context(patch.object(files_route, "S3Client", lambda: _FakeS3()))
        self.stack.enter_context(patch.object(files_route, "IngestionPipeline", _FakeIngestionPipeline))
        self.stack.enter_context(patch.object(files_route, "async_session_factory", _fake_session_factory))

        self.stack.enter_context(patch.object(ingestion_route, "IngestionPipeline", _FakeIngestionPipeline))
        self.stack.enter_context(patch.object(ingestion_route, "DocumentRepository", _FakeDocumentRepository))
        self.stack.enter_context(patch.object(ingestion_route, "async_session_factory", _fake_session_factory))

        self.stack.enter_context(patch.object(query_route, "get_workflow", lambda: fake_workflow))
        async def fake_build_document_info(reranked):
            return {}, {}

        self.stack.enter_context(patch.object(query_route, "_build_document_info", fake_build_document_info))
        self.stack.enter_context(patch.object(query_route, "DocumentRepository", _FakeDocumentRepository))
        self.stack.enter_context(patch.object(query_route, "async_session_factory", _fake_session_factory))

        @asynccontextmanager
        async def fake_service_in_session():
            yield fake_service

        self.stack.enter_context(patch.object(vs_route, "_service_in_session", fake_service_in_session))

        self.client = TestClient(api_main.app, raise_server_exceptions=False)
        self.addCleanup(self.client.close)

    def test_health_and_static_routes(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/util").status_code, 200)
        self.assertEqual(self.client.get("/playground").status_code, 200)

    def test_files_routes(self):
        response = self.client.get("/files?limit=10000&purpose=assistants")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["object"], "list")
        self.assertEqual(response.json()["data"][0]["status"], "uploaded")

        upload = self.client.post(
            "/files",
            files={"file": ("sample.txt", b"hello", "text/plain")},
            data={"purpose": "assistants"},
        )
        self.assertEqual(upload.status_code, 200)
        self.assertEqual(upload.json()["object"], "file")

        single = self.client.get("/files/file-1")
        self.assertEqual(single.status_code, 200)
        self.assertEqual(single.json()["id"], "file-1")

        content = self.client.get("/files/file-1/content")
        self.assertEqual(content.status_code, 200)
        self.assertEqual(content.content, b"hello world")

        deleted = self.client.delete("/files/file-1")
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(deleted.json(), {"id": "file-1", "deleted": True, "object": "file"})

    def test_ingestion_routes(self):
        text = self.client.post("/ingest/text", data={"text": "hello", "title": "Doc"})
        self.assertEqual(text.status_code, 200)
        self.assertEqual(text.json()["document_id"], "doc-ingest-text")

        file = self.client.post(
            "/ingest/file",
            files={"file": ("sample.txt", b"hello", "text/plain")},
            data={"title": "Doc"},
        )
        self.assertEqual(file.status_code, 200)
        self.assertEqual(file.json()["document_id"], "doc-ingest-file")

        both = self.client.post("/ingest", data={"text": "hello", "title": "Doc"})
        self.assertEqual(both.status_code, 200)
        self.assertEqual(both.json()["document_id"], "doc-ingest-text")

        chunks = self.client.get("/chunks/doc-ingest-text")
        self.assertEqual(chunks.status_code, 200)
        self.assertEqual(chunks.json()["document_id"], "doc-ingest-text")

        stats = self.client.get("/index/stats")
        self.assertEqual(stats.status_code, 200)
        self.assertEqual(stats.json(), {"qdrant_count": 0, "bm25_count": 0})

    def test_query_route(self):
        response = self.client.post("/query", json={"query": "What is this?"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "Test answer")

    def test_vector_store_routes(self):
        create = self.client.post("/vector_stores", json={"name": "store"})
        self.assertEqual(create.status_code, 200)
        store_id = create.json()["id"]

        listed = self.client.get("/vector_stores")
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.json()["data"][0]["id"], store_id)

        got = self.client.get(f"/vector_stores/{store_id}")
        self.assertEqual(got.status_code, 200)

        modified = self.client.post(f"/vector_stores/{store_id}", json={"name": "renamed"})
        self.assertEqual(modified.status_code, 200)

        attached = self.client.post(f"/vector_stores/{store_id}/files", json={"file_id": "doc-1"})
        self.assertEqual(attached.status_code, 200)

        files = self.client.get(f"/vector_stores/{store_id}/files")
        self.assertEqual(files.status_code, 200)
        self.assertEqual(files.json()["data"][0]["id"], "file-1")

        file = self.client.get(f"/vector_stores/{store_id}/files/file-1")
        self.assertEqual(file.status_code, 200)

        updated = self.client.post(f"/vector_stores/{store_id}/files/file-1", json={"attributes": {"a": 1}})
        self.assertEqual(updated.status_code, 200)

        content = self.client.get(f"/vector_stores/{store_id}/files/file-1/content")
        self.assertEqual(content.status_code, 200)

        batch = self.client.post(f"/vector_stores/{store_id}/file_batches", json={"file_ids": ["doc-1"]})
        self.assertEqual(batch.status_code, 200)

        batch_get = self.client.get(f"/vector_stores/{store_id}/file_batches/vsfb-1")
        self.assertEqual(batch_get.status_code, 200)

        batch_files = self.client.get(f"/vector_stores/{store_id}/file_batches/vsfb-1/files")
        self.assertEqual(batch_files.status_code, 200)

        batch_cancel = self.client.post(f"/vector_stores/{store_id}/file_batches/vsfb-1/cancel")
        self.assertEqual(batch_cancel.status_code, 200)

        search = self.client.post(f"/vector_stores/{store_id}/search", json={"query": "hello"})
        self.assertEqual(search.status_code, 200)
        self.assertEqual(search.json()["data"][0]["filename"], "alpha.pdf")

        removed_file = self.client.delete(f"/vector_stores/{store_id}/files/file-1")
        self.assertEqual(removed_file.status_code, 200)

        deleted = self.client.delete(f"/vector_stores/{store_id}")
        self.assertEqual(deleted.status_code, 200)

    def test_vector_store_validation_errors(self):
        bad_order = self.client.get("/vector_stores/vs-1/files?order=sideways")
        self.assertEqual(bad_order.status_code, 400)
        missing = self.client.get("/vector_stores/missing")
        self.assertEqual(missing.status_code, 404)


if __name__ == "__main__":
    unittest.main()
