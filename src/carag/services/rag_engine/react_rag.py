from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent
from typing_extensions import TypedDict

from carag.core.constants import get_constants
from carag.services.document_ingestion.ingest import DocumentIngestionEngine
from carag.services.rag_engine.detect_mal_intent import call_guard_api
from carag.services.rag_engine.hallucination_check import has_hallucinations

constants = get_constants()


class AgentState(TypedDict):
    """Enhanced state for the ReAct agent with security and hallucination tracking."""

    messages: List[Dict[str, Any]]
    question: str
    tenant_id: str
    retrieved_documents: List[Document]  # Track all retrieved documents
    hallucination_result: Dict[str, Any]  # Store hallucination check results
    is_malicious: bool  # Track malicious intent


def get_react_agent(tenant_id: str):
    engine = DocumentIngestionEngine(tenant_id=tenant_id)

    # Track documents retrieved during tool execution
    retrieved_docs_tracker = []

    @tool
    def search_vector_db(query: str) -> List[Document]:
        """Tool to search the vector database"""
        docs = engine.search_within_tenant(query)
        retrieved_docs_tracker.extend(docs)  # Track retrieved documents
        return docs

    def route_malintent(state: AgentState):
        """Check if the initial question contains malicious content."""
        # Only check on the first message
        if len(state["messages"]) == 1:  # Only user message present
            api_result = call_guard_api(state["question"])
            if api_result["is_malicious"]:
                return "malicious"
        return "benign"

    def malicious_response(state: AgentState):
        """Handle malicious content by returning a safe response."""
        return {
            "messages": state["messages"]
            + [
                {
                    "role": "assistant",
                    "content": "Cannot process this request as it appears to contain malicious content or attempts to manipulate the system.",
                }
            ],
            "is_malicious": True,
            "retrieved_documents": [],
            "hallucination_result": {},
        }

    def run_agent(state: AgentState):
        """Run the ReAct agent normally for benign queries."""
        # Reset the document tracker for this run
        retrieved_docs_tracker.clear()

        # Run the standard ReAct agent
        base_agent = create_react_agent(
            constants.llm,
            tools=[search_vector_db],
            checkpointer=InMemorySaver(),
            state_schema=AgentState,
        )

        # Invoke the agent with current state
        result = base_agent.invoke(state)

        # Extract the final assistant message
        final_message = None
        for msg in reversed(result["messages"]):
            if msg["role"] == "assistant":
                final_message = msg
                break

        return {
            "messages": result["messages"],
            "retrieved_documents": retrieved_docs_tracker.copy(),  # Store retrieved docs
            "is_malicious": False,
        }

    def check_hallucinations(state: AgentState):
        """Check the final answer for hallucinations based on retrieved documents."""
        if state["is_malicious"] or not state["retrieved_documents"]:
            return {
                "hallucination_result": {
                    "skipped": True,
                    "reason": "malicious_content"
                    if state["is_malicious"]
                    else "no_documents_retrieved",
                }
            }

        # Extract the final assistant message
        final_answer = ""
        for msg in reversed(state["messages"]):
            if msg["role"] == "assistant":
                final_answer = msg["content"]
                break

        if not final_answer:
            return {
                "hallucination_result": {
                    "skipped": True,
                    "reason": "no_answer_generated",
                }
            }

        # Perform hallucination detection

        chunks = [doc.page_content for doc in state["retrieved_documents"]]
        hallucination_result = has_hallucinations(
            state["question"], final_answer, chunks
        )

        return {"hallucination_result": hallucination_result}

    def finalize_response(state: AgentState):
        """Final step to prepare the response with security and hallucination info."""
        # Extract the final assistant message
        final_answer = ""
        for msg in reversed(state["messages"]):
            if msg["role"] == "assistant":
                final_answer = msg["content"]
                break

        # Prepare response with metadata
        response_data = {
            "answer": final_answer,
            "is_malicious": state.get("is_malicious", False),
            "retrieved_documents_count": len(state.get("retrieved_documents", [])),
            "hallucination_check": state.get("hallucination_result", {}),
        }

        # Add warning if hallucination was detected
        if (
            not state.get("is_malicious", False)
            and state.get("hallucination_result", {}).get("skipped") != True
            and state.get("hallucination_result", {}).get("hallucination_ratio", 0) > 0
        ):
            hallucination_ratio = state["hallucination_result"]["hallucination_ratio"]
            if hallucination_ratio > 0.5:
                severity = "high"
                warning = "⚠️ High likelihood of hallucinations detected. Please verify critical information."
            elif hallucination_ratio > 0.2:
                severity = "medium"
                warning = (
                    "⚠️ Some unsupported information detected. Verify important details."
                )
            else:
                severity = "low"
                warning = "ℹ️ Minor inconsistencies detected."

            response_data["hallucination_warning"] = {
                "has_warning": True,
                "severity": severity,
                "message": warning,
                "hallucination_ratio": hallucination_ratio,
            }

        return response_data

    # Build the enhanced agent graph
    graph_builder = StateGraph(AgentState)

    # Add nodes
    graph_builder.add_node("malicious_response", malicious_response)
    graph_builder.add_node("run_agent", run_agent)
    graph_builder.add_node("check_hallucinations", check_hallucinations)
    graph_builder.add_node("finalize", finalize_response)

    # Set conditional entry point
    graph_builder.set_conditional_entry_point(
        route_malintent, {"malicious": "malicious_response", "benign": "run_agent"}
    )

    # Add edges
    graph_builder.add_edge("malicious_response", "check_hallucinations")
    graph_builder.add_edge("run_agent", "check_hallucinations")
    graph_builder.add_edge("check_hallucinations", "finalize")
    graph_builder.add_edge("finalize", END)

    return graph_builder.compile()
