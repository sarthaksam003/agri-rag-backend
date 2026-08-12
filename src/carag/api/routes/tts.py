from fastapi import APIRouter, Depends
from fastapi.responses import Response

from carag.api.dependencies import get_tenant_id
from carag.api.schemas import ErrorResponse, TTSRequest
from carag.services.bhasini import synthesize

router = APIRouter()


@router.post(
    "/synthesize",
    responses={
        200: {"content": {"audio/wav": {}}},
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def text_to_speech(
    request: TTSRequest,
    tenant_id: str = Depends(get_tenant_id),
) -> Response:
    """
    Lazy TTS — converts text to speech on demand.
    Called when user clicks the speaker button.
    Returns raw audio bytes.
    """
    audio_bytes = await synthesize(
        text=request.text,
        target_language=request.language,
        gender=request.gender,
    )

    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": "inline; filename=speech.wav",
        },
    )
