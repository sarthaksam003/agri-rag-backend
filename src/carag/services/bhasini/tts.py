import base64

from loguru import logger

from carag.services.bhasini.client import call_inference
from carag.services.bhasini.schemas import TaskType


async def synthesize(
    text: str,
    target_language: str,
    gender: str = "female",
) -> bytes:
    """Convert text to audio bytes using Bhasini TTS."""
    payload = {
        "pipelineTasks": [
            {
                "taskType": "tts",
                "config": {
                    "language": {"sourceLanguage": target_language},
                    "gender": gender,
                },
            }
        ],
        "inputData": {
            "input": [{"source": text}],
        },
    }

    result = await call_inference(
        task_type=TaskType.TTS,
        source_language=target_language,
        target_language=None,
        payload=payload,
    )

    audio_b64 = result["pipelineResponse"][0]["audio"][0]["audioContent"]
    audio_bytes = base64.b64decode(audio_b64)
    logger.info(f"TTS result ({target_language}): {len(audio_bytes)} bytes")
    return audio_bytes
