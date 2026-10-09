import os
from functools import lru_cache
from typing import TYPE_CHECKING

import redis
# from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_qdrant import FastEmbedSparse
from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler
from qdrant_client import QdrantClient
from carag.services.rag_engine.hf_embeddings import HuggingFaceEmbeddings
from carag.core.config import get_settings

# from carag.services.document_ingestion.chunker import (
#     CustomTableAwareChunker,
#     custom_table_aware_chunker,
# )

if TYPE_CHECKING:
    pass

settings = get_settings()
print("cwd:", os.getcwd())

_ = Langfuse(
    public_key=settings.LF_PUBLIC_KEY,
    secret_key=settings.LF_SECRET_KEY,
    host=settings.LF_HOST,
)
langfuse = get_client()


class Constants:
    vectorstore: QdrantClient = QdrantClient(
        f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}"
    )
    llm: BaseChatModel = ChatOpenAI(
        base_url=settings.OPENAI_API_BASE,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
        model=settings.OPENAI_API_MODEL,
        temperature=0.3,
    )
    print("EMBEDDINGS_API_KEY:", repr(
        settings.EMBEDDINGS_API_KEY.get_secret_value()))
    print("EMBEDDINGS_API_BASE:", repr(settings.EMBEDDINGS_API_BASE))
    print("EMBEDDINGS_API_MODEL:", repr(settings.EMBEDDINGS_API_MODEL))
    embedder: Embeddings = HuggingFaceEmbeddings(
        model=settings.EMBEDDINGS_API_MODEL,
        api_key=settings.HF_TOKEN.get_secret_value(),
    )

    # from carag.services.rag_engine.rerank import VllmRerank

    # reranker: VllmRerank = VllmRerank(
    #     base_url=settings.RERANKER_API_BASE,
    #     api_key=settings.RERANKER_API_KEY.get_secret_value(),
    #     model=settings.RERANKER_API_MODEL,
    #     top_n=5,
    # )
    from carag.services.rag_engine.rerank import MixedbreadRerank

    reranker: MixedbreadRerank = MixedbreadRerank(
        base_url=settings.RERANKER_API_BASE,
        api_key=settings.RERANKER_API_KEY,
        model=settings.RERANKER_API_MODEL,
        top_n=5,
    )
    from carag.services.document_ingestion.mineruloader import MineruLoader

    loader: type[MineruLoader] = MineruLoader

    sparse_embedder: FastEmbedSparse = FastEmbedSparse(
        model_name="Qdrant/bm25",
        cache_dir=os.path.expanduser("~/.cache/huggingface/hub"),
    )

    observability_handler: BaseCallbackHandler = CallbackHandler()
    redis_client: redis.Redis = redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_CHAT_DB,
        password=settings.REDIS_AUTH,
        decode_responses=False,
        socket_timeout=2,
        socket_connect_timeout=2,
    )


@lru_cache
def get_constants() -> Constants:
    return Constants()
