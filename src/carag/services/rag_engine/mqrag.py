import hashlib
from collections import defaultdict
from typing import Annotated, Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph import END, START, StateGraph, add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from carag.core.constants import get_constants
from carag.services.document_ingestion.ingest import DocumentIngestionEngine

# from carag.services.rag_engine.detect_mal_intent import call_guard_api
# from carag.services.rag_engine.hallucination_check import (
#     detect_hallucinations,
# )
from carag.services.rag_engine.prompt import mqtemplate, query_prompt

constants = get_constants()
llm = constants.llm
reranker = constants.reranker


class QueryContextDocument(BaseModel):
    """Enhanced document with query context"""

    document: Document
    source_queries: list[str]  # Which queries retrieved this document
    retrieval_score: float | None = None


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    generated_queries: list[str]
    context: list[QueryContextDocument]  # Changed to use enhanced documents
    answer: str
    tenant_id: str
    hallucination_result: dict[str, Any]
    query_to_docs_mapping: dict[str, list[Document]]


class QueryOutput(BaseModel):
    """Generated query variations for improved document retrieval."""

    questions: list[str] = Field(
        description="List of 1-6 alternative query formulations based on complexity"
    )


# def route_malintent(state: State):
#     """Check if the question contains malicious content."""
#     api_result = call_guard_api(state["question"])
#     if api_result["is_malicious"]:
#         return "malicious"
#     elif not api_result["is_malicious"]:
#         return "benign"
#


def generate_queries(state: State):
    """Generate multiple query variations based on the original question."""
    messages = query_prompt.invoke({"question": state["question"]})

    # Use structured output with the Pydantic model
    structured_llm = llm.with_structured_output(QueryOutput)
    response = structured_llm.invoke(messages)

    # Extract queries from the structured response
    queries = response.questions

    # Always include the original query
    all_queries = [state["question"]] + queries

    return {"generated_queries": all_queries}


def retrieve_with_query_context(state: State):
    """Retrieve documents for all generated queries and track query context."""
    engine = DocumentIngestionEngine(tenant_id=state["tenant_id"])

    # Track documents and which queries retrieved them
    doc_to_queries_map = defaultdict(list)
    query_to_docs_map = defaultdict(list)
    all_docs = []
    seen_hashes = set()

    # Retrieve documents for each query and track provenance
    for query in state["generated_queries"]:
        retrieved_docs = engine.search_within_tenant(
            query=query, k=15
        )  # Get more docs per query

        for doc in retrieved_docs:
            # Create hash of page_content for deduplication
            content_hash = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()

            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                all_docs.append(doc)
                doc_to_queries_map[content_hash].append(query)
                query_to_docs_map[query].append(doc)
            else:
                # Document already seen, just add the query to the provenance
                doc_to_queries_map[content_hash].append(query)
                # Find the existing document and update its query mapping
                for existing_doc in all_docs:
                    existing_hash = hashlib.md5(
                        existing_doc.page_content.encode("utf-8")
                    ).hexdigest()
                    if existing_hash == content_hash:
                        # This document is already in the list, just update the query mapping
                        break

    # Rerank the deduplicated documents
    reranked_docs = reranker.compress_documents(all_docs, state["question"])

    # Create QueryContextDocument objects with query provenance
    context_docs = []
    for doc in reranked_docs:
        content_hash = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
        source_queries = doc_to_queries_map[content_hash]

        context_doc = QueryContextDocument(
            document=doc,
            source_queries=source_queries,
            retrieval_score=getattr(doc.metadata, "relevance_score", None),
        )
        context_docs.append(context_doc)

    return {"context": context_docs, "query_to_docs_mapping": dict(query_to_docs_map)}


def generate_with_query_context(state: State):
    """Generate answer using the retrieved context with query provenance."""
    # Format documents with enhanced context including query information
    formatted_chunks = []
    for idx, context_doc in enumerate(state["context"], start=1):
        doc = context_doc.document
        source_queries = context_doc.source_queries

        # Create enhanced chunk with query context
        chunk = f"""CHUNK ID: {idx}
Retrieved by queries: {", ".join(source_queries)}
Retrieval score: {context_doc.retrieval_score or "N/A"}
Metadata: {doc.metadata}
Content:
{doc.page_content}
--------------"""
        formatted_chunks.append(chunk)

    docs_content = "\n\n".join(formatted_chunks)

    # Build conversation history from messages
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

    messages = mqtemplate.invoke(
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
        "generated_queries": state.get("generated_queries", [state["question"]]),
    }


# def check_for_hallucinations(state: State):
#     """Check if the generated answer contains hallucinations based on retrieved context."""
#     # Extract just the text content from documents
#     chunks = [ctx_doc.document.page_content for ctx_doc in state["context"]]
#
#     # Detect hallucinations in the generated answer
#     result = detect_hallucinations(state["question"], state["answer"], chunks)
#
#     return {"hallucination_result": result}


# def cannot_generate(state: State):
#     """Handle malicious content by returning a safe response."""
#     return {
#         "answer": "Cannot process this request as it appears to contain malicious content or attempts to manipulate the system.",
#         "hallucination_result": {},  # Empty result for malicious content
#     }


# Build the graph with enhanced query context
graph_builder = StateGraph(State)

# Add all nodes
# graph_builder.add_node("cannot_generate", cannot_generate)
graph_builder.add_node("generate_queries", generate_queries)
graph_builder.add_node("retrieve", retrieve_with_query_context)  # Updated
graph_builder.add_node("generate", generate_with_query_context)  # Updated
# graph_builder.add_node("check_hallucinations", check_for_hallucinations)

# Set conditional entry point for malicious intent detection
# graph_builder.set_conditional_entry_point(
#     route_malintent, {"malicious": "cannot_generate", "benign": "generate_queries"}
# )

# Add the sequence for benign queries
graph_builder.add_edge(START, "generate_queries")
graph_builder.add_edge("generate_queries", "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", END)
# graph_builder.add_edge("generate", "check_hallucinations")
# graph_builder.add_edge("check_hallucinations", END)

# Add edge for malicious content
# graph_builder.add_edge("cannot_generate", END)

# Compile the graph
mqgraph = graph_builder.compile()
