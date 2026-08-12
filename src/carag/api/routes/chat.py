import uuid
from datetime import datetime

from fastapi import APIRouter, Depends

from carag.api.dependencies import get_tenant_id
from carag.api.schemas import (
    ChatMessageRequest,
    ChatMessageResponse,
    ErrorResponse,
)
from carag.services.chat_service import run_rag_pipeline

router = APIRouter()


@router.post(
    "/message",
    response_model=ChatMessageResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def send_chat_message(
    request: ChatMessageRequest,
    tenant_id: str = Depends(get_tenant_id),
) -> ChatMessageResponse:
    """Send a text message to the RAG chatbot."""
    result = await run_rag_pipeline(
        message=request.message,
        session_id=request.session_id,
        tenant_id=tenant_id,
    )

    return ChatMessageResponse(
        id=str(uuid.uuid4()),
        session_id=result.session_id,
        message=result.user_message,
        response=result.response,
        timestamp=datetime.now(),
        chunks=result.chunks,
        hallucination_spans=result.hallucination_result,
        retrieved_documents_count=result.retrieved_documents_count,
    )
