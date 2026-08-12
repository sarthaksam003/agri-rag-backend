import time

import httpx
from loguru import logger

from carag.core.config import get_settings
from carag.services.bhasini.schemas import PipelineConfig, TaskType

settings = get_settings()

# Module-level cache for pipeline configs
_pipeline_cache: dict[str, tuple[PipelineConfig, float]] = {}


async def _get_pipeline_config(
    task_type: TaskType,
    source_language: str,
    target_language: str | None = None,
) -> PipelineConfig:
    """
    Step 1 of ULCA flow: call getModelsPipeline to discover
    the inference callback URL and serviceId for a given task.
    Caches the result.
    """
    cache_key = f"{task_type}:{source_language}:{target_language or ''}"

    # Check cache
    if cache_key in _pipeline_cache:
        config, timestamp = _pipeline_cache[cache_key]
        if time.time() - timestamp < settings.BHASINI_PIPELINE_CACHE_TTL:
            return config

    # Build pipeline task config
    task_config: dict = {"language": {"sourceLanguage": source_language}}
    if target_language:
        task_config["language"]["targetLanguage"] = target_language

    payload = {
        "pipelineTasks": [
            {
                "taskType": task_type.value,
                "config": task_config,
            }
        ],
        "pipelineRequestConfig": {
            "pipelineId": "64392f96daac500b55c543cd",  # Bhashini open pipeline
        },
    }

    headers = {
        "ulcaApiKey": settings.BHASINI_ULCA_API_KEY,
        "userID": settings.BHASINI_ULCA_USER_ID,
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            settings.BHASINI_ULCA_PIPELINE_URL,
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()

    logger.debug(f"ULCA pipeline response for {task_type}: {data}")

    # Extract callback URL and auth
    inference_api = data["pipelineInferenceAPIEndPoint"]
    callback_url = inference_api["callbackUrl"]
    inference_api_key = inference_api.get("inferenceApiKey", {}).get("value")

    # Extract service ID from response config
    response_config = data.get("pipelineResponseConfig", [{}])[0]
    service_id = response_config.get("config", [{}])[0].get("serviceId")

    config = PipelineConfig(
        callback_url=callback_url,
        inference_api_key=inference_api_key,
    )

    # Set the right service ID field
    if task_type == TaskType.ASR:
        config.asr_service_id = service_id
    elif task_type == TaskType.TTS:
        config.tts_service_id = service_id
    elif task_type == TaskType.TRANSLATION:
        config.translation_service_id = service_id

    # Cache it
    _pipeline_cache[cache_key] = (config, time.time())

    logger.info(
        f"Cached ULCA pipeline config for {task_type} "
        f"({source_language} → {target_language}): serviceId={service_id}"
    )

    return config


async def call_inference(
    task_type: TaskType,
    source_language: str,
    target_language: str | None,
    payload: dict,
) -> dict:
    """
    Step 2 of ULCA flow: call the inference callback URL
    with the actual task payload.
    """
    config = await _get_pipeline_config(task_type, source_language, target_language)

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if config.inference_api_key:
        headers["Authorization"] = config.inference_api_key

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            config.callback_url,
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        return response.json()
