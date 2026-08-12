import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from loguru import logger

from carag.api.history import conversation_history
from carag.core.constants import get_constants

constants = get_constants()


class RagMode(str, Enum):
    SIMPLE = "simple"
    MULTIQUERY = "multiquery"


@dataclass
class RetrievalDetail:
    chunk_id: int
    source_queries: list[str]
    retrieval_score: float | None
    metadata: dict[str, Any]


@dataclass
class RagResult:
    session_id: str
    user_message: str
    response: str
    chunks: list[str]
    hallucination_result: dict[str, Any] | None
    retrieved_documents_count: int
    # Multiquery-specific fields
    mode: RagMode = RagMode.SIMPLE
    generated_queries: list[str] = field(default_factory=list)
    retrieval_analysis: dict[str, Any] = field(default_factory=dict)
    chunk_retrieval_info: list[RetrievalDetail] = field(default_factory=list)


def _get_graph(mode: RagMode):
    """Lazy import to avoid circular deps and loading both graphs on startup."""
    if mode == RagMode.SIMPLE:
        from carag.services.rag_engine.simple_rag import graph

        return graph
    elif mode == RagMode.MULTIQUERY:
        from carag.services.rag_engine.mqrag import mqgraph

        return mqgraph


def _extract_simple_result(rag_result: dict) -> tuple[list[str], dict]:
    """Extract chunks from simple RAG output."""
    docs = rag_result.get("context", [])
    chunks = [d.page_content if hasattr(d, "page_content") else str(d) for d in docs]
    return chunks, {}


def _extract_multiquery_result(rag_result: dict) -> tuple[list[str], dict]:
    """Extract chunks and retrieval info from multiquery RAG output."""
    context_docs = rag_result.get("context", [])
    generated_queries = rag_result.get("generated_queries", [])
    query_to_docs_mapping = rag_result.get("query_to_docs_mapping", {})

    docs = [ctx_doc.document for ctx_doc in context_docs]
    chunks = [d.page_content if hasattr(d, "page_content") else str(d) for d in docs]

    # Build retrieval details
    chunk_retrieval_info = []
    for idx, ctx_doc in enumerate(context_docs):
        detail = RetrievalDetail(
            chunk_id=idx + 1,
            source_queries=ctx_doc.source_queries,
            retrieval_score=ctx_doc.retrieval_score,
            metadata=ctx_doc.document.metadata,
        )
        chunk_retrieval_info.append(detail)

    retrieval_analysis = {
        "total_queries_generated": len(generated_queries),
        "queries_with_results": len([q for q, d in query_to_docs_mapping.items() if d]),
        "retrieval_strategy": "multiquery_with_provenance",
    }

    extra = {
        "generated_queries": generated_queries,
        "retrieval_analysis": retrieval_analysis,
        "chunk_retrieval_info": chunk_retrieval_info,
    }

    return chunks, extra


async def run_rag_pipeline(
    message: str,
    session_id: str | None,
    tenant_id: str,
    mode: RagMode = RagMode.SIMPLE,
) -> RagResult:
    """
    Pure text-in, text-out RAG pipeline.
    Supports both simple and multiquery modes.
    Used by text chat, multiquery chat, and voice chat routes.
    """
    session_id = session_id or str(uuid.uuid4())
    history_key = f"{session_id}||{tenant_id}"

    # Get conversation history
    messages_history = conversation_history.get(history_key, [])

    # Add user message
    user_message = HumanMessage(content=message)
    messages_history.append(user_message)

    logger.info(
        f"Processing message (mode={mode.value}) for tenant: {tenant_id}, "
        f"session: {session_id}, history length: {len(messages_history)}"
    )

    # Get the right graph and invoke
    graph = _get_graph(mode)
    rag_result = graph.invoke(
        {
            "messages": messages_history,
            "question": message,
            "tenant_id": tenant_id,
        },
        config={"callbacks": [constants.observability_handler]},
    )

    # Extract response
    ai_response = rag_result["answer"]

    # Update history
    ai_message = AIMessage(content=ai_response)
    messages_history.append(ai_message)
    conversation_history[history_key] = messages_history

    # Trim history
    if len(messages_history) > 20:
        conversation_history[history_key] = messages_history[-20:]

    # Extract chunks based on mode
    if mode == RagMode.SIMPLE:
        chunks, extra = _extract_simple_result(rag_result)
    else:
        chunks, extra = _extract_multiquery_result(rag_result)

    return RagResult(
        session_id=session_id,
        user_message=message,
        response=ai_response,
        chunks=chunks,
        hallucination_result=rag_result.get("hallucination_result"),
        retrieved_documents_count=len(chunks),
        mode=mode,
        generated_queries=extra.get("generated_queries", []),
        retrieval_analysis=extra.get("retrieval_analysis", {}),
        chunk_retrieval_info=extra.get("chunk_retrieval_info", []),
    )
