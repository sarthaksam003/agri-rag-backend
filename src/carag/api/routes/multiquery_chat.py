import uuid
from datetime import datetime
import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from carag.api.dependencies import get_knowledge_base_tenant_id
from carag.api.schemas import (
    ErrorResponse,
    MultiqueryChatMessageRequest,
    MultiqueryChatMessageResponse,
    RetrievalInfo,
)
from carag.services.chat_service import (
    RagGenerationInProgressError,
    RagMode,
    run_rag_pipeline,
    stream_rag_pipeline,
)
from carag.auth.manager import current_active_user
from carag.auth.models import User

router = APIRouter()


@router.post(
    "/message",
    response_model=MultiqueryChatMessageResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def send_multiquery_chat_message(
    request: MultiqueryChatMessageRequest,
    tenant_id: str = Depends(get_knowledge_base_tenant_id),
    user: User = Depends(current_active_user),
) -> MultiqueryChatMessageResponse:
    """Send a message using Multiquery RAG approach."""
    result = await run_rag_pipeline(
        message=request.message,
        session_id=request.session_id,
        tenant_id=tenant_id,
        user_id=str(user.id),
        max_queries=request.max_queries,
        mode=RagMode.MULTIQUERY,
        display_user_message=request.message,
        display_language=request.source_language,
        input_type="text",
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


@router.post(
    "/message/stream",
    responses={
        400: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def stream_multiquery_chat_message(
    request: MultiqueryChatMessageRequest,
    tenant_id: str = Depends(get_knowledge_base_tenant_id),
    user: User = Depends(current_active_user),
):
    """Stream a text response using Multiquery RAG."""

    async def event_generator():
        try:
            async for event in stream_rag_pipeline(
                message=request.message,
                session_id=request.session_id,
                tenant_id=tenant_id,
                user_id=str(user.id),
                max_queries=request.max_queries,
                mode=RagMode.MULTIQUERY,
                display_user_message=request.message,
                display_language=request.source_language,
                input_type="text",
            ):
                yield json.dumps(event) + "\n"

        except RagGenerationInProgressError as exc:
            yield json.dumps(
                {
                    "type": "error",
                    "status": 409,
                    "detail": str(exc),
                }
            ) + "\n"

        except Exception as exc:
            yield json.dumps(
                {
                    "type": "error",
                    "status": 500,
                    "detail": str(exc),
                }
            ) + "\n"

    return StreamingResponse(
        event_generator(),
        media_type="application/x-ndjson",
    )
