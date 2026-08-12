from enum import Enum

from pydantic import BaseModel


class TaskType(str, Enum):
    ASR = "asr"
    TTS = "tts"
    TRANSLATION = "translation"


class PipelineTask(BaseModel):
    taskType: TaskType
    config: dict


class PipelineRequest(BaseModel):
    pipelineTasks: list[PipelineTask]
    pipelineRequestConfig: dict


class PipelineConfig(BaseModel):
    """Cached result from ULCA pipeline discovery."""

    callback_url: str
    inference_api_key: str | None
    asr_service_id: str | None = None
    tts_service_id: str | None = None
    translation_service_id: str | None = None


class ASRInput(BaseModel):
    audioContent: str  # base64
    source_language: str


class TTSInput(BaseModel):
    text: str
    target_language: str
    gender: str = "female"


class TranslationInput(BaseModel):
    text: str
    source_language: str
    target_language: str
