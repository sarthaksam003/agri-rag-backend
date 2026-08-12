from collections.abc import Sequence
from typing_extensions import override
import requests
from langchain.retrievers.document_compressors.base import (
    BaseDocumentCompressor,
)
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field, SecretStr


class RerankResult(BaseModel):
    index: int
    relevance_score: float


# from langchain_voyageai import VoyageAIRerank
class VllmRerank(BaseDocumentCompressor):
    base_url: str = Field(..., description="Base URL of the vLLM reranker API")
    api_key: SecretStr = Field(..., description="API key for the vLLM reranker API")
    model: str = Field(..., description="The name of the model to use for reranking")
    top_n: int = Field(default=5, description="The number of documents to return.")

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        if not documents:
            return []
        texts_to_rerank = [doc.page_content for doc in documents]

        payload = {
            "model": self.model,
            "query": query,
            "documents": texts_to_rerank,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
        }

        try:
            response = requests.post(
                f"{self.base_url}/rerank", headers=headers, json=payload
            )
            response.raise_for_status()  # Raise an exception for bad status codes
            results_json = response.json()["results"]
            reranked_results = [RerankResult(**res) for res in results_json]

        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error calling reranker API: {e}")

        final_documents: Sequence[Document] = []
        for result in reranked_results:
            original_doc = documents[result.index]
            original_doc.metadata["relevance_score"] = result.relevance_score
            final_documents.append(original_doc)

        return final_documents[: self.top_n]

    @override
    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        raise ValueError("Async mode not supported")
