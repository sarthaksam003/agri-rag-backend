from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException

from carag.auth.manager import current_active_user
from carag.auth.models import User
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
    user: User = Depends(current_active_user),
) -> SuccessResponse:
    """
    Get chat history for a specific session.
    """
    owner_id = str(user.id)
    session_key = f"{session_id}||{owner_id}"

    if session_key not in conversation_history:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found",
        )

    messages_history = conversation_history.get(
        session_key,
        [],
    )

    # Convert to simple format for response
    history_data = []

    for i in range(0, len(messages_history), 2):
        if i + 1 >= len(messages_history):
            continue

        user_message = messages_history[i]
        assistant_message = messages_history[i + 1]

        user_kwargs = user_message.additional_kwargs
        assistant_kwargs = assistant_message.additional_kwargs

        history_data.append(
            {
                "user": user_kwargs.get(
                    "display_content",
                    user_message.content,
                ),
                "assistant": assistant_kwargs.get(
                    "display_content",
                    assistant_message.content,
                ),
                "user_timestamp": user_kwargs.get("timestamp"),
                "assistant_timestamp": assistant_kwargs.get("timestamp"),
                "user_language": user_kwargs.get(
                    "display_language",
                    "en",
                ),
                "assistant_language": assistant_kwargs.get(
                    "display_language",
                    "en",
                ),
                "user_input_type": user_kwargs.get(
                    "input_type",
                    "text",
                ),
                "sources": assistant_kwargs.get("sources", []),
            }
        )
    return SuccessResponse(
        message="Chat history retrieved successfully",
        data={
            "session_id": session_id,
            "title": conversation_history.get_title(
                session_id,
                owner_id,
            ),
            "history": history_data,
            "total_messages": len(messages_history),
        },
    )


@router.delete("/sessions/{session_id}", response_model=SuccessResponse)
async def delete_chat_session(
    session_id: str,
    user: User = Depends(current_active_user),
) -> SuccessResponse:
    """
    Delete a specific chat session belonging to the current user.
    """

    owner_id = str(user.id)
    session_key = f"{session_id}||{owner_id}"

    if session_key not in conversation_history:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found",
        )

    del conversation_history[session_key]

    return SuccessResponse(
        message=f"Session {session_id} deleted successfully"
    )


@router.get("/sessions", response_model=SuccessResponse)
async def list_chat_sessions(
    user: User = Depends(current_active_user),
    limit: int = 20,
    offset: int = 0,
) -> SuccessResponse:
    """
    List chat sessions for the current tenant,
    ordered by most recent interaction.
    """
    owner_id = str(user.id)
    all_sessions = conversation_history.list_sessions(
        owner_id
    )

    total = len(all_sessions)

    paginated_sessions = all_sessions[
        offset: offset + limit
    ]

    session_data = []

    for session_id in paginated_sessions:
        messages = conversation_history.get(
            f"{session_id}||{owner_id}",
            [],
        )

        last_updated_timestamp = (
            conversation_history.get_last_updated(
                session_id,
                owner_id,
            )
        )

        session_data.append(
            {
                "session_id": session_id,
                "title": (
                    conversation_history.get_title(
                        session_id,
                        owner_id,
                    )
                    or "Untitled conversation"
                ),
                "message_count": len(messages),
                "last_updated": (
                    datetime.fromtimestamp(
                        last_updated_timestamp
                    )
                    if last_updated_timestamp is not None
                    else None
                ),
            }
        )

    return SuccessResponse(
        message="Chat sessions retrieved successfully",
        data={
            "sessions": session_data,
            "total": total,
            "limit": limit,
            "offset": offset,
        },
    )
