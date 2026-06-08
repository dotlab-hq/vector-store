from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class DocumentModel(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    source_path: Mapped[str] = mapped_column(String, nullable=False)
    source_type: Mapped[str] = mapped_column(String, nullable=False)
    author: Mapped[str] = mapped_column(String, default="")
    department: Mapped[str] = mapped_column(String, default="")
    tags: Mapped[str] = mapped_column(Text, default="")  # JSON-encoded
    metadata_json: Mapped[str] = mapped_column("metadata", Text, default="{}")
    s3_key: Mapped[str | None] = mapped_column(String, nullable=True)
    bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    mime_type: Mapped[str] = mapped_column(
        String(64), default="application/octet-stream"
    )
    content_text: Mapped[str] = mapped_column(
        Text, default=""
    )  # raw text for text-only docs (no S3)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    chunks: Mapped[list["ChunkModel"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class ChunkModel(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    document_id: Mapped[str] = mapped_column(
        String, ForeignKey("documents.id"), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    parent_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("chunks.id"), nullable=True
    )
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    position: Mapped[int] = mapped_column(Integer, default=0)
    section: Mapped[str] = mapped_column(String, default="")
    entities: Mapped[str] = mapped_column(Text, default="")  # JSON-encoded
    metadata_json: Mapped[str] = mapped_column("metadata", Text, default="{}")
    vector_store_id: Mapped[str | None] = mapped_column(
        String, nullable=True, index=True
    )
    attributes_json: Mapped[str] = mapped_column(Text, default="{}")

    document: Mapped["DocumentModel"] = relationship(back_populates="chunks")
    children: Mapped[list["ChunkModel"]] = relationship(back_populates="parent")
    parent: Mapped["ChunkModel | None"] = relationship(
        back_populates="children", remote_side="ChunkModel.id"
    )


class VectorStoreModel(Base):
    __tablename__ = "vector_stores"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False, default="")
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="in_progress"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    last_active_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now()
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    expires_after_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    chunking_strategy: Mapped[str] = mapped_column(String(16), default="auto")
    chunk_size_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_overlap_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    file_counts_json: Mapped[str] = mapped_column(
        Text,
        default='{"in_progress":0,"completed":0,"cancelled":0,"failed":0,"total":0}',
    )
    usage_bytes: Mapped[int] = mapped_column(BigInteger, default=0)

    files: Mapped[list["VectorStoreFileModel"]] = relationship(
        back_populates="vector_store", cascade="all, delete-orphan"
    )
    batches: Mapped[list["VectorStoreFileBatchModel"]] = relationship(
        back_populates="vector_store", cascade="all, delete-orphan"
    )


class VectorStoreFileModel(Base):
    __tablename__ = "vector_store_files"
    __table_args__ = (
        UniqueConstraint(
            "vector_store_id", "source_document_id", name="uq_vs_file_doc"
        ),
        Index("ix_vs_file_status_next", "status", "next_attempt_at"),
        Index("ix_vs_file_batch", "batch_id"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    vector_store_id: Mapped[str] = mapped_column(
        String, ForeignKey("vector_stores.id", ondelete="CASCADE"), nullable=False
    )
    source_document_id: Mapped[str] = mapped_column(
        String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    object: Mapped[str] = mapped_column(String(32), default="vector_store.file")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    next_attempt_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    locked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    locked_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    attributes_json: Mapped[str] = mapped_column(Text, default="{}")
    batch_id: Mapped[str | None] = mapped_column(
        String,
        ForeignKey("vector_store_file_batches.id", ondelete="SET NULL"),
        nullable=True,
    )

    vector_store: Mapped["VectorStoreModel"] = relationship(back_populates="files")


class VectorStoreFileBatchModel(Base):
    __tablename__ = "vector_store_file_batches"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    vector_store_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("vector_stores.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    object: Mapped[str] = mapped_column(String(32), default="vector_store.file_batch")
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="in_progress"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    file_counts_json: Mapped[str] = mapped_column(
        Text,
        default='{"in_progress":0,"completed":0,"cancelled":0,"failed":0,"total":0}',
    )
    attributes_json: Mapped[str] = mapped_column(Text, default="{}")

    vector_store: Mapped["VectorStoreModel"] = relationship()


class ProcessingTaskModel(Base):
    """PostgreSQL-backed job queue. API writes rows; worker polls and processes."""

    __tablename__ = "processing_tasks"
    __table_args__ = (
        Index(
            "ix_task_status_priority_next",
            "status",
            "priority",
            "next_attempt_at",
        ),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    task_type: Mapped[str] = mapped_column(String(64), nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, default="{}")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    priority: Mapped[int] = mapped_column(Integer, default=0)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, default=5)
    locked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    locked_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    next_attempt_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
