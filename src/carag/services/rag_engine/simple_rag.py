from typing import Annotated, Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph import END, START, StateGraph, add_messages
from loguru import logger
from markdownify import markdownify as md
from typing_extensions import TypedDict

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
    last_question = state["question"]
    engine = DocumentIngestionEngine(tenant_id=state["tenant_id"])
    retrieved_docs = engine.search_within_tenant(query=state["question"], k=20)
    markdownified_docs = [
        Document(
            page_content=md(doc.page_content, heading_style="ATX"),
            metadata=doc.metadata,
        )
        for doc in retrieved_docs
    ]
    logger.info(markdownified_docs)
    reranked_docs = reranker.compress_documents(
        markdownified_docs,
        state["question"],
    )
    return {"context": reranked_docs}


def generate(state: State):
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

    messages = template.invoke(
        {
            "question": state["question"],
            "context": docs_content,
            "conversation_history": conversation_history,
        }
    )

    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "messages": [
            HumanMessage(content=state["question"]),
            AIMessage(content=response.content),
        ],
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
    print(graph.invoke({"question": "What is e4m3?", "tenant_id": "demo-tenant"}))
