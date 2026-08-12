from datetime import datetime

from fastapi import APIRouter, Depends

from carag.api.dependencies import get_tenant_id
from carag.api.history import conversation_history
from carag.api.schemas import SuccessResponse

router = APIRouter()


@router.get("/health", response_model=SuccessResponse)
async def sessions_health_check(
    tenant_id: str = Depends(get_tenant_id),
) -> SuccessResponse:
    """
    Health check for sessions endpoint.
    TODO: Implement session management with PostgreSQL
    """
    return SuccessResponse(message="Sessions endpoint is healthy")


# TODO: Implement session management endpoints
# - POST /sessions - Create new session


@router.get("/sessions/{session_id}/history", response_model=SuccessResponse)
async def get_chat_history(
    session_id: str,
    tenant_id: str = Depends(get_tenant_id),
) -> SuccessResponse:
    """
    Get chat history for a specific session.
    """
    messages_history = conversation_history.get(f"{session_id}||{tenant_id}", [])

    # Convert to simple format for response
    history_data = []
    for i in range(0, len(messages_history), 2):
        if i + 1 < len(messages_history):
            history_data.append(
                {
                    "user": messages_history[i].content,
                    "assistant": messages_history[i + 1].content,
                }
            )

    return SuccessResponse(
        message="Chat history retrieved successfully",
        data={
            "session_id": session_id,
            "history": history_data,
            "total_messages": len(messages_history),
        },
    )


@router.delete("/sessions/{session_id}", response_model=SuccessResponse)
async def delete_chat_session(
    session_id: str,
    tenant_id: str = Depends(get_tenant_id),
) -> SuccessResponse:
    """
    Delete a specific chat session and its history.
    """

    if f"{session_id}||{tenant_id}" in conversation_history:
        del conversation_history[f"{session_id}||{tenant_id}"]

    return SuccessResponse(message=f"Session {session_id} deleted successfully")


@router.get("/sessions", response_model=SuccessResponse)
async def list_chat_sessions(
    tenant_id: str = Depends(get_tenant_id),
    limit: int = 20,
    offset: int = 0,
) -> SuccessResponse:
    """
    List chat sessions for the current tenant.
    """
    sessions = [
        s.split("||", 1)[0]
        for s in conversation_history.keys()
        if s.endswith(f"||{tenant_id}")
    ][offset : offset + limit]

    session_data = []
    for session_id in sessions:
        messages = conversation_history[f"{session_id}||{tenant_id}"]
        session_data.append(
            {
                "session_id": session_id,
                "message_count": len(messages),
                "last_updated": datetime.now(),
            }
        )
    tenant_keys = [
        k for k in conversation_history.keys() if k.endswith(f"||{tenant_id}")
    ]
    total = len(tenant_keys)
    sessions = [k.split("||", 1)[0] for k in tenant_keys][offset : offset + limit]
    return SuccessResponse(
        message="Chat sessions retrieved successfully",
        data={
            "sessions": session_data,
            "total": total,
            "limit": limit,
            "offset": offset,
        },
    )
