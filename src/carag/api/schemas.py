from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field

class RagConfigurationUpdate(BaseModel):
    rag_mode: Literal["simple", "multiquery"]
    max_queries: int = Field(
        ge=2,
        le=6,
    )
    
# Chat Schemas
class ChatMessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = None
    source_language: str = Field(default="en")
    regenerate_from_index: int | None = Field(default=None, ge=0)

class SourceInfo(BaseModel):
    source: str
    document_id: str | None = None
    page_number: int | None = None
    chunk_id: str | None = None
    chunk_index: int | None = None
    content: str
    relevance_score: float | None = None



class ChatMessageResponse(BaseModel):
    id: str
    session_id: str
    session_title: str | None = None
    message: str
    response: str
    timestamp: datetime
    chunks: list[str]
    sources: list[SourceInfo] = Field(default_factory=list)
    hallucination_spans: dict[str, Any] | None = None
    retrieved_documents_count: int = 0


class ChatSession(BaseModel):
    id: str
    title: str
    created_at: datetime
    message_count: int
    last_message_at: datetime | None = None


# Document Schemas
class DocumentIngestionRequest(BaseModel):
    metadata: dict[str, Any] | None = None


class DocumentIngestionResponse(BaseModel):
    document_id: str
    filename: str
    status: str  # processing, completed, failed
    chunks_processed: int = 0
    total_chunks: int = 0
    processing_time: float | None = None
    error_message: str | None = None
    job_id: str | None = None


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    file_size: int
    ingested_at: datetime
    chunk_count: int
    status: str


class BatchDocumentIngestionResponse(BaseModel):
    processed_files: list[DocumentIngestionResponse]
    total_processed: int
    total_failed: int


class DeleteDocumentResponse(BaseModel):
    message: str
    deleted_chunks: int
    document_id: str


class DeleteTenantResponse(BaseModel):
    message: str
    deleted_chunks: int
    tenant_id: str


# Generic Response Schemas
class ErrorResponse(BaseModel):
    error: str
    details: dict[str, Any] | None = None


class SuccessResponse(BaseModel):
    message: str
    data: dict[str, Any] | None = None


# class MultiqueryChatMessageRequest(ChatMessageRequest):
#     """Extended request for multiquery RAG with additional options"""

#     use_multiquery: bool = Field(default=True)
#     max_queries: int = Field(default=6, ge=1, le=6)


# class RetrievalInfo(BaseModel):
#     """Information about how a chunk was retrieved"""

#     chunk_id: int
#     source_queries: list[str]
#     retrieval_score: float | None = None
#     metadata: dict[str, Any]


# class MultiqueryChatMessageResponse(ChatMessageResponse):
#     """Extended response for multiquery RAG"""

#     generated_queries: list[str] = Field(default_factory=list)
#     retrieval_strategy: str = "multiquery"
#     retrieval_analysis: dict[str, Any] = Field(default_factory=dict)
#     chunk_retrieval_info: list[RetrievalInfo] = Field(default_factory=list)


# Voice Chat Schemas
class VoiceChatRequest(BaseModel):
    """Received as form data, not JSON — audio comes as UploadFile."""

    session_id: str | None = None
    source_language: str = Field(
        default="hi", description="Language of the audio input"
    )


class VoiceChatResponse(BaseModel):
    id: str
    session_id: str
    session_title: str | None = None
    original_message: str  # what ASR heard (in source language)
    # translated to RAG language (if applicable)
    translated_message: str | None
    response: str  # RAG response (in RAG language)
    translated_response: (
        str | None
    )  # translated back to source language (if applicable)
    source_language: str
    timestamp: datetime
    chunks: list[str]
    hallucination_spans: dict[str, Any] | None = None
    retrieved_documents_count: int = 0

class TranscriptionResponse(BaseModel):
    text: str
    source_language: str
    
# TTS Schemas
class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: str = Field(default="hi")
    gender: str = Field(default="female")
