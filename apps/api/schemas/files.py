from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class FileObject(BaseModel):
    id: str
    bytes: int
    created_at: int
    filename: str
    object: Literal["file"] = "file"
    purpose: Literal[
        "assistants",
        "assistants_output",
        "batch",
        "batch_output",
        "fine-tune",
        "fine-tune-results",
        "vision",
        "user_data",
    ] = "assistants"
    status: Literal["uploaded", "processing", "processed", "error"] = "uploaded"
    expires_at: int | None = None
    status_details: str | None = None


class FileDeletedResponse(BaseModel):
    id: str
    deleted: bool = True
    object: Literal["file"] = "file"


class FileListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[FileObject] = Field(default_factory=list)
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool = False
