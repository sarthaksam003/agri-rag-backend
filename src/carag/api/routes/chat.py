import uuid
import json
from datetime import datetime
from carag.auth.manager import current_active_user
from carag.auth.models import User
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from carag.services.bhasini import translate
from carag.core.config import get_settings
from carag.api.dependencies import get_knowledge_base_tenant_id
from carag.api.schemas import (
    ChatMessageRequest,
    ChatMessageResponse,
    ErrorResponse,
    SourceInfo,
)
from loguru import logger
from carag.services.chat_service import (
    RagGenerationInProgressError,
    run_rag_pipeline,
    update_last_assistant_display_for_user,
    stream_rag_pipeline
)
router = APIRouter()

settings = get_settings()


@router.post(
    "/message",
    response_model=ChatMessageResponse,
    responses={
        400: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },)
async def send_chat_message(
    request: ChatMessageRequest,
    tenant_id: str = Depends(get_knowledge_base_tenant_id),
    user: User = Depends(current_active_user),
) -> ChatMessageResponse:
    """Send a text message to the RAG chatbot."""
    rag_input = request.message

    if request.source_language != settings.BHASINI_SOURCE_LANGUAGE:
        rag_input = await translate(
            text=request.message,
            source_language=request.source_language,
            target_language=settings.BHASINI_SOURCE_LANGUAGE,
        )

    try:
        result = await run_rag_pipeline(
            message=rag_input,
            session_id=request.session_id,
            tenant_id=tenant_id,
            user_id=str(user.id),
            display_user_message=request.message,
            display_language=request.source_language,
            input_type="text",
        )
    except RagGenerationInProgressError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    response_text = result.response

    if request.source_language != settings.BHASINI_SOURCE_LANGUAGE:
        response_text = await translate(
            text=result.response,
            source_language=settings.BHASINI_SOURCE_LANGUAGE,
            target_language=request.source_language,
        )
    update_last_assistant_display_for_user(
        session_id=result.session_id,
        user_id=str(user.id),
        display_content=response_text,
        display_language=request.source_language,
    )
    return ChatMessageResponse(
        id=str(uuid.uuid4()),
        session_id=result.session_id,
        session_title=result.session_title,
        message=rag_input,
        response=response_text,
        timestamp=datetime.now(),
        chunks=result.chunks,
        sources=_build_sources(result),
        hallucination_spans=result.hallucination_result,
        retrieved_documents_count=result.retrieved_documents_count,
    )


@router.post(
    "/message/stream",
    responses={
        400: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def stream_chat_message(
    request: ChatMessageRequest,
    tenant_id: str = Depends(get_knowledge_base_tenant_id),
    user: User = Depends(current_active_user),
):
    """Stream a text RAG response to the client."""

    rag_input = request.message

    if request.source_language != settings.BHASINI_SOURCE_LANGUAGE:
        rag_input = await translate(
            text=request.message,
            source_language=request.source_language,
            target_language=settings.BHASINI_SOURCE_LANGUAGE,
        )

    async def event_generator():
        try:
            async for event in stream_rag_pipeline(
                message=rag_input,
                session_id=request.session_id,
                tenant_id=tenant_id,
                user_id=str(user.id),
                display_user_message=request.message,
                display_language=request.source_language,
                input_type="text",
                regenerate_from_index=request.regenerate_from_index,
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
            logger.exception("Streaming RAG request failed")
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


def _build_sources(result) -> list[SourceInfo]:
    sources = []

    for index, detail in enumerate(result.retrieval_details):
        metadata = detail.metadata

        content = (
            result.chunks[index]
            if index < len(result.chunks)
            else ""
        )

        sources.append(
            SourceInfo(
                source=str(
                    metadata.get(
                        "original_filename",
                        metadata.get("source", "Unknown"),
                    )
                ),
                chunk_id=(
                    str(metadata["chunk_id"])
                    if metadata.get("chunk_id") is not None
                    else None
                ),
                chunk_index=metadata.get("chunk_index"),
                content=content,
                relevance_score=detail.relevance_score,
            )
        )

    return sources
