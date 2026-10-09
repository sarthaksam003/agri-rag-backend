from __future__ import annotations
import time
import os
from typing import List
from loguru import logger
from huggingface_hub import InferenceClient
from langchain_core.embeddings import Embeddings


class HuggingFaceEmbeddings(Embeddings):
    """LangChain-compatible embeddings backed by Hugging Face Inference API."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("HF_TOKEN")

        if not self.api_key:
            raise ValueError(
                "HF_TOKEN is required for Hugging Face embeddings.")

        self.client = InferenceClient(
            provider="hf-inference",
            api_key=self.api_key,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []

        for text in texts:
            result = self.client.feature_extraction(
                text,
                model=self.model,
            )

            embeddings.append(result.tolist())

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        start_time = time.perf_counter()

        result = self.client.feature_extraction(
            f"Query: {text}",
            model=self.model,
        )

        elapsed = time.perf_counter() - start_time

        logger.info(
            f"[TIMING] Query embedding: {elapsed:.3f}s"
        )

        return result.tolist()