from __future__ import annotations

import time
from pathlib import Path
from typing import List

from langchain_core.embeddings import Embeddings
from loguru import logger
from sentence_transformers import SentenceTransformer


class HuggingFaceEmbeddings(Embeddings):
    """LangChain-compatible embeddings using a locally stored SentenceTransformer."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        model_path: str | Path | None = None,
        batch_size: int = 4,
        max_seq_length: int = 2048,
    ) -> None:
        # `model` remains accepted for compatibility with the existing constants
        # configuration (for example, "Snowflake/snowflake-arctic-embed-l-v2.0").
        # Inference is local; no HF_TOKEN or hosted API call is required.
        self.model_name = model
        project_root = Path(__file__).resolve().parents[4]

        if model_path is not None:
            resolved_model_path = Path(model_path).expanduser()
        else:
            model_as_path = Path(model).expanduser()
            if model_as_path.is_dir():
                resolved_model_path = model_as_path
            else:
                model_dir_name = model.rstrip("/\\").split("/")[-1]
                resolved_model_path = (
                    project_root / "models" / model_dir_name
                )

        self.model_path = resolved_model_path.resolve()
        if not self.model_path.is_dir():
            raise FileNotFoundError(
                f"Local embedding model directory not found: {self.model_path}"
            )

        self.batch_size = batch_size
        self.max_seq_length = max_seq_length

        self.model = SentenceTransformer(
            str(self.model_path),
            local_files_only=True,
        )
        self.model.max_seq_length = self.max_seq_length

        logger.info(
            f"Embedding configuration: batch_size={self.batch_size}, "
            f"max_seq_length={self.max_seq_length}"
        )
        logger.info(f"Loaded local embedding model from: {self.model_path}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        start_time = time.perf_counter()

        # Preserve the query prefix used by the previous hosted implementation.
        result = self.model.encode(
            f"Query: {text}",
            batch_size=1,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        elapsed = time.perf_counter() - start_time
        logger.info(f"[TIMING] Query embedding: {elapsed:.3f}s")

        return result.tolist()
