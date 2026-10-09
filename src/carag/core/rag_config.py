from typing import Literal

from beanie import Document
from pydantic import Field


class RagConfiguration(Document):
    """
    Global RAG configuration for the shared AgriChat knowledge base.
    """

    rag_mode: Literal["simple", "multiquery"] = "simple"

    max_queries: int = Field(
        default=2,
        ge=2,
        le=6,
    )

    class Settings:
        name = "rag_configuration"