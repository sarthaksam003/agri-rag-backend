import base64

from loguru import logger

from carag.services.bhasini.client import call_inference
from carag.services.bhasini.schemas import TaskType


async def transcribe(audio_bytes: bytes, source_language: str) -> str:
    """Convert audio bytes to text using Bhasini ASR."""
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    payload = {
        "pipelineTasks": [
            {
                "taskType": "asr",
                "config": {
                    "language": {"sourceLanguage": source_language},
                    "audioFormat": "wav",
                    "samplingRate": 16000,
                },
            }
        ],
        "inputData": {
            "audio": [{"audioContent": audio_b64}],
        },
    }

    result = await call_inference(
        task_type=TaskType.ASR,
        source_language=source_language,
        target_language=None,
        payload=payload,
    )

    text = result["pipelineResponse"][0]["output"][0]["source"]
    logger.info(f"ASR result ({source_language}): {text[:100]}...")
    return text
