from typing import Annotated, Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph import END, START, StateGraph, add_messages
from loguru import logger
from markdownify import markdownify as md
from typing_extensions import TypedDict
import time
from carag.core.constants import get_constants
from carag.services.document_ingestion.ingest import DocumentIngestionEngine

# from carag.services.rag_engine.detect_mal_intent import call_guard_api
# from carag.services.rag_engine.hallucination_check import detect_hallucinations
from carag.services.rag_engine.prompt import template

constants = get_constants()
llm = constants.llm
reranker = constants.reranker


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    context: list[Document]
    answer: str
    tenant_id: str
    hallucination_result: dict[str, Any]
    response_language: str

# def route_malintent(state: State):
#     question = state["question"]
#     api_result = call_guard_api(question)
#     if api_result["is_malicious"]:
#         return "malicious"
#
#     elif not api_result["is_malicious"]:
#         return "benign"
#


def retrieve(state: State):
    retrieval_start = time.perf_counter()
    last_question = state["question"]
    engine_start = time.perf_counter()

    engine = DocumentIngestionEngine(tenant_id=state["tenant_id"])

    search_start = time.perf_counter()

    retrieved_docs = engine.search_within_tenant(
        query=state["question"],
        k=20,
    )

    search_elapsed = time.perf_counter() - search_start
    engine_elapsed = time.perf_counter() - engine_start
    logger.info(
        f"[TIMING] DocumentIngestionEngine initialization: "
        f"{engine_elapsed:.3f}s"
    )
    logger.info(
        f"[TIMING] Document search: {search_elapsed:.3f}s "
        f"({len(retrieved_docs)} documents)"
    )

    markdownified_docs = [
        Document(
            page_content=md(doc.page_content, heading_style="ATX"),
            metadata=doc.metadata,
        )
        for doc in retrieved_docs
    ]

    logger.info(markdownified_docs)

    rerank_start = time.perf_counter()

    reranked_docs = reranker.compress_documents(
        markdownified_docs,
        state["question"],
    )

    rerank_elapsed = time.perf_counter() - rerank_start
    retrieval_elapsed = time.perf_counter() - retrieval_start

    logger.info(
        f"[TIMING] Reranking: {rerank_elapsed:.3f}s "
        f"({len(reranked_docs)} documents)"
    )

    logger.info(
        f"[TIMING] Retrieval pipeline total: {retrieval_elapsed:.3f}s"
    )

    return {"context": reranked_docs}


async def generate(state: State):
    formatted_chunks = []
    for idx, doc in enumerate(state["context"], start=1):
        chunk = f"""CHUNK ID: {idx}
                Metadata: {doc.metadata}
                Content:
                {doc.page_content}
                --------------"""
        formatted_chunks.append(chunk)

    docs_content = "\n\n".join(formatted_chunks)

    conversation_history = ""
    for msg in state["messages"]:
        if hasattr(msg, "type"):
            if msg.type == "human":
                conversation_history += f"User: {msg.content}\n"
            elif msg.type == "ai":
                conversation_history += f"Assistant: {msg.content}\n"
            else:
                if msg.get("role") == "user":
                    conversation_history += f"User: {msg.get('content', '')}\n"
                elif msg.get("role") == "assistant" or msg.get("role") == "ai":
                    conversation_history += f"Assistant: {msg.get('content', '')}\n"
    prompt_start = time.perf_counter()
    logger.info(
        f"[LANG DEBUG] RAG response_language={state.get('response_language')}"
    )
    messages = template.invoke(
        {
            "question": state["question"],
            "context": docs_content,
            "conversation_history": conversation_history,
            "response_language": state["response_language"],
        }
    )
    prompt_elapsed = time.perf_counter() - prompt_start
    logger.info(
        f"[TIMING] Prompt construction: {prompt_elapsed:.3f}s"
    )
    response_content = ""

    llm_start = time.perf_counter()
    first_token_time = None

    async for chunk in llm.astream(messages):
        if chunk.content:
            if first_token_time is None:
                first_token_time = time.perf_counter()

                logger.info(
                    f"[TIMING] LLM time to first token: "
                    f"{first_token_time - llm_start:.3f}s"
                )

            response_content += chunk.content

    llm_elapsed = time.perf_counter() - llm_start

    logger.info(
        f"[TIMING] LLM generation total: {llm_elapsed:.3f}s"
    )
    return {
        "answer": response_content,
    }


# def check_for_hallucinations(state: State):
#     """Check if the generated answer contains hallucinations based on retrieved context."""
#
#     # Extract just the text content from documents
#     chunks = [doc.page_content for doc in state["context"]]
#
#     # Detect hallucinations in the generated answer
#     result = detect_hallucinations(state["question"], state["answer"], chunks)
#
#     return {"hallucination_result": result}


# def cannot_generate(state: State):
#     err_msg = "Cannot process this request as it appears to contain malicious content or attempts to manipulate the system."
#     return {
#         "answer": err_msg,
#         "messages": [
#             HumanMessage(content=state["question"]),
#             AIMessage(content=err_msg),
#         ],
#     }


graph_builder = StateGraph(State)


# graph_builder.add_node("cannot_generate", cannot_generate)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)
# graph_builder.add_node("check_hallucinations", check_for_hallucinations)
# graph_builder.set_conditional_entry_point(
#     route_malintent, {"malicious": "cannot_generate", "benign": "retrieve"}
# )
graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", END)
# graph_builder.add_edge("generate", "check_hallucinations")  # Add this edge
# graph_builder.add_edge("check_hallucinations", END)  # Then end
# graph_builder.add_edge("cannot_generate", END)
graph = graph_builder.compile()


if __name__ == "__main__":
    print(graph.invoke(
        {"question": "What is e4m3?", "tenant_id": "demo-tenant"}))
