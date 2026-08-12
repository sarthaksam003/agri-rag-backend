from loguru import logger

from carag.services.bhasini.client import call_inference
from carag.services.bhasini.schemas import TaskType


async def translate(
    text: str,
    source_language: str,
    target_language: str,
) -> str:
    """Translate text using Bhasini NMT."""
    if source_language == target_language:
        return text

    payload = {
        "pipelineTasks": [
            {
                "taskType": "translation",
                "config": {
                    "language": {
                        "sourceLanguage": source_language,
                        "targetLanguage": target_language,
                    },
                },
            }
        ],
        "inputData": {
            "input": [{"source": text}],
        },
    }

    result = await call_inference(
        task_type=TaskType.TRANSLATION,
        source_language=source_language,
        target_language=target_language,
        payload=payload,
    )

    translated = result["pipelineResponse"][0]["output"][0]["target"]
    logger.info(
        f"Translation ({source_language}→{target_language}): "
        f"'{text[:50]}...' → '{translated[:50]}...'"
    )
    return translated
