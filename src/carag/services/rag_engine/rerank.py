from collections.abc import Sequence
import time
from loguru import logger
import requests
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from typing_extensions import override

RERANK_MAX_CHARS = 4000


class RerankResult(BaseModel):
    index: int
    relevance_score: float


class MixedbreadRerank(BaseDocumentCompressor):
    """
    Reranker backed by the Mixedbread hosted Reranking API.

    The RAG pipeline remains independent of the inference provider because
    this class exposes LangChain's BaseDocumentCompressor interface.
    """

    base_url: str = Field(
        ...,
        description="Base URL of the Mixedbread reranking API",
    )
    api_key: SecretStr = Field(
        ...,
        description="API key for the Mixedbread reranking API",
    )
    model: str = Field(
        ...,
        description="The name of the reranking model to use",
    )
    top_n: int = Field(
        default=5,
        description="The number of documents to return",
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:

        if not documents:
            return []

        texts_to_rerank = [
            doc.page_content[:RERANK_MAX_CHARS]
            for doc in documents
        ]

        payload = {
            "model": self.model,
            "query": query,
            "documents": texts_to_rerank,
            # "top_k": self.top_n,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": (
                f"Bearer {self.api_key.get_secret_value()}"
            ),
        }

        try:
            request_start = time.perf_counter()

            response = requests.post(
                f"{self.base_url}/reranking",
                headers=headers,
                json=payload,
                timeout=60,
            )

            request_elapsed = time.perf_counter() - request_start

            logger.info(
                f"[TIMING] Mixedbread HTTP request: "
                f"{request_elapsed:.3f}s | "
                f"docs={len(texts_to_rerank)} | "
                f"chars={sum(len(text) for text in texts_to_rerank)}"
            )

            response.raise_for_status()

            json_start = time.perf_counter()

            response_data = response.json()

            json_elapsed = time.perf_counter() - json_start

            logger.info(
                f"[TIMING] Mixedbread response JSON parsing: "
                f"{json_elapsed:.3f}s"
            )

            reranked_results = [
                RerankResult(
                    index=result["index"],
                    relevance_score=result["score"],
                )
                for result in response_data["data"]
            ]

        except requests.exceptions.RequestException as e:
            raise ValueError(
                f"Error calling Mixedbread reranking API: {e}"
            ) from e

        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(
                f"Unexpected response from Mixedbread reranking API: {e}"
            ) from e

        final_documents: list[Document] = []

        for result in reranked_results:
            original_doc = documents[result.index]

            original_doc.metadata["relevance_score"] = (
                result.relevance_score
            )

            final_documents.append(original_doc)

        return final_documents[: self.top_n]

    @override
    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        texts_to_rerank = [
            doc.page_content[:RERANK_MAX_CHARS]
            for doc in documents
        ]
        raise ValueError("Async mode not supported")
