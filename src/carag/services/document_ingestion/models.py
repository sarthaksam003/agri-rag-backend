import uuid
from datetime import datetime
from typing import Any

from langchain_qdrant import SparseVector
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Represents a single chunk of a document"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None
    sparse_embedding: SparseVector | None = None


class IngestedDocument(BaseModel):
    """Represents a document after ingestion processing"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_filename: str
    file_path: str
    file_size: int
    mime_type: str = "application/pdf"
    chunks: list[DocumentChunk] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    processed_at: datetime = Field(default_factory=datetime.now)
    processing_time: float | None = None
