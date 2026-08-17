from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[3]
ENV_FILE = BASE_DIR / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )
    QDRANT_HOST: str = ""
    QDRANT_PORT: str = ""
    QDRANT_COLLECTION_NAME: str = ""
    QDRANT_TENANT_FIELD: str = ""

    OPENAI_API_KEY: SecretStr = SecretStr("")
    OPENAI_API_BASE: str = ""
    OPENAI_API_MODEL: str = ""
    OPENAI_API_GPU_UTIL: float = 0.0
    OPENAI_API_PORT: int = 0

    EMBEDDINGS_API_KEY: SecretStr = SecretStr("")
    EMBEDDINGS_API_BASE: str = ""
    EMBEDDINGS_API_MODEL: str = ""
    EMBEDDINGS_API_DIMENSION: int = 0
    EMBEDDINGS_API_PORT: int = 0
    EMBEDDINGS_API_GPU_UTIL: float = 0.0

    FRONTEND_URL: str = "http://localhost:5173"

    RERANKER_API_KEY: SecretStr = SecretStr("")
    RERANKER_API_BASE: str = ""
    RERANKER_API_MODEL: str = ""
    RERANKER_API_DIMENSION: int = 0
    RERANKER_API_PORT: int = 0
    RERANKER_API_GPU_UTIL: float = 0.0

    PDFMD_API_BASE: str = ""
    PDFMD_API_GPU_UTIL: float = 0.0
    PDFMD_API_PORT: int = 0
    PDFMD_DATA_RAW: str = ""
    PDFMD_DATA_PROCESSED: str = ""
    PDFMD_DATA_TMP: str = ""

    CHUNKER_CHUNK_SIZE: int = 0

    GUARD_API_BASE: str = ""
    HALLUCINATION_API_BASE: str = ""

    LF_PUBLIC_KEY: str = ""
    LF_SECRET_KEY: str = ""
    LF_HOST: str = ""

    REDIS_HOST: str = Field(default="", validation_alias="RED_HOST")
    REDIS_PORT: int = Field(default=0, validation_alias="RED_PORT")
    REDIS_CHAT_DB: int = Field(default=0, validation_alias="RED_CHAT_DB")
    REDIS_AUTH: str = Field(default="", validation_alias="RED_AUTH")
    RED_AUTH_DB: int = 3

    CELERY_REDIS_DB: int = 2

    MONGO_AUTH_PORT: int = 0
    MONGO_AUTH_HOST: str = ""
    MONGO_AUTH_USERNAME: str = ""
    MONGO_AUTH_PASSWD: str = ""
    MONGO_AUTH_DB: str = "carag_auth"
    AUTH_SECRET: str

    #GOOGLE OAUTH SETTINGS
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: SecretStr = SecretStr("")
    OAUTH_STATE_SECRET: str

    #callback URL
    BACKEND_URL: str = "http://localhost:8000"

    # CHUNKER SETTINGS
    USE_AGENTIC_TAGGER: bool = False
    TABLE_LEN_CHUNK_SIZE_FACTOR: int = 2
    CHUNK_TEXT_DIVIDER_FACTOR: int = 2
    TAGGER_N_CHUNKS_PER_BATCH: int = 5

    # Bhasini ULCA
    BHASINI_ULCA_API_KEY: str = ""
    BHASINI_ULCA_USER_ID: str = ""
    BHASINI_ULCA_PIPELINE_URL: str = (
        "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline"
    )
    BHASINI_SOURCE_LANGUAGE: str = "en"  # your RAG engine's language
    BHASINI_PIPELINE_CACHE_TTL: int = 3600  # seconds, pipeline config cache


@lru_cache
def get_settings() -> Settings:
    return Settings()
