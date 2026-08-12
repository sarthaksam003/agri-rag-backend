import uuid
from datetime import datetime

from fastapi import APIRouter, Depends

from carag.api.dependencies import get_tenant_id
from carag.api.schemas import (
    ErrorResponse,
    MultiqueryChatMessageRequest,
    MultiqueryChatMessageResponse,
    RetrievalInfo,
)
from carag.services.chat_service import RagMode, run_rag_pipeline

router = APIRouter()


@router.post(
    "/message",
    response_model=MultiqueryChatMessageResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def send_multiquery_chat_message(
    request: MultiqueryChatMessageRequest,
    tenant_id: str = Depends(get_tenant_id),
) -> MultiqueryChatMessageResponse:
    """Send a message using Multiquery RAG approach."""
    result = await run_rag_pipeline(
        message=request.message,
        session_id=request.session_id,
        tenant_id=tenant_id,
        mode=RagMode.MULTIQUERY,
    )

    # Convert RetrievalDetail to RetrievalInfo for API schema
    chunk_retrieval_info = [
        RetrievalInfo(
            chunk_id=detail.chunk_id,
            source_queries=detail.source_queries,
            retrieval_score=detail.retrieval_score,
            metadata=detail.metadata,
        )
        for detail in result.chunk_retrieval_info
    ]

    return MultiqueryChatMessageResponse(
        id=str(uuid.uuid4()),
        session_id=result.session_id,
        message=result.user_message,
        response=result.response,
        timestamp=datetime.now(),
        chunks=result.chunks,
        hallucination_spans=result.hallucination_result,
        retrieved_documents_count=result.retrieved_documents_count,
        generated_queries=result.generated_queries,
        retrieval_strategy="multiquery_with_provenance",
        retrieval_analysis=result.retrieval_analysis,
        chunk_retrieval_info=chunk_retrieval_info,
    )
