import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, Form, UploadFile
from loguru import logger
from carag.auth.manager import current_active_user
from carag.auth.models import User
from carag.api.dependencies import get_knowledge_base_tenant_id
from carag.api.schemas import (
    ErrorResponse, VoiceChatResponse, TranscriptionResponse)
from carag.core.config import get_settings
from carag.services.bhasini import transcribe, translate
from carag.services.chat_service import (
    # RagMode,
    run_rag_pipeline,
    update_last_assistant_display_for_user,
)
router = APIRouter()
settings = get_settings()


@router.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def transcribe_audio(
    audio: UploadFile,
    source_language: str = Form(default="hi"),
    user: User = Depends(current_active_user),
) -> TranscriptionResponse:
    """
    Convert uploaded audio to text using Bhashini ASR.
    Does not run translation or RAG.
    """
    audio_bytes = await audio.read()

    logger.info(
        f"Transcription request: {len(audio_bytes)} bytes, "
        f"language: {source_language}"
    )

    text = await transcribe(audio_bytes, source_language)

    return TranscriptionResponse(
        text=text,
        source_language=source_language,
    )


@router.post(
    "/message",
    response_model=VoiceChatResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def voice_chat_message(
    audio: UploadFile,
    source_language: str = Form(default="hi"),
    session_id: str | None = Form(default=None),
    tenant_id: str = Depends(get_knowledge_base_tenant_id),
    user: User = Depends(current_active_user),
) -> VoiceChatResponse:
    """
    Voice message to RAG chatbot.
    Audio in → ASR → translate → RAG → translate back → text out.

    The active RAG mode and query count are controlled
    by the server-side RAG configuration.
    """
    rag_language = settings.BHASINI_SOURCE_LANGUAGE
    needs_translation = source_language != rag_language
    # mode = RagMode(rag_mode)

    # Step 1: ASR
    audio_bytes = await audio.read()
    logger.info(
        f"Voice input: {len(audio_bytes)} bytes, "
        f"language: {source_language}"
    )
    user_text = await transcribe(audio_bytes, source_language)

    # Step 2: Translate to RAG language (if needed)
    translated_message = None
    rag_input = user_text
    if needs_translation:
        translated_message = await translate(
            text=user_text,
            source_language=source_language,
            target_language=rag_language,
        )
        rag_input = translated_message

    # Step 3: RAG pipeline
    result = await run_rag_pipeline(
        message=rag_input,
        session_id=session_id,
        tenant_id=tenant_id,
        user_id=str(user.id),
        display_user_message=user_text,
        display_language=source_language,
        input_type="voice",
    )

    # Step 4: Translate response back (if needed)
    translated_response = None
    if needs_translation:
        translated_response = await translate(
            text=result.response,
            source_language=rag_language,
            target_language=source_language,
        )
    display_response = (
        translated_response
        if translated_response is not None
        else result.response
    )

    update_last_assistant_display_for_user(
        session_id=result.session_id,
        user_id=str(user.id),
        display_content=display_response,
        display_language=source_language,
    )
    return VoiceChatResponse(
        id=str(uuid.uuid4()),
        session_id=result.session_id,
        session_title=result.session_title,
        original_message=user_text,
        translated_message=translated_message,
        response=result.response,
        translated_response=translated_response,
        source_language=source_language,
        timestamp=datetime.now(),
        chunks=result.chunks,
        hallucination_spans=result.hallucination_result,
        retrieved_documents_count=result.retrieved_documents_count,
    )
