from __future__ import annotations

import json
import time
from typing import Any

from langchain_core.messages import (
    BaseMessage,
    messages_from_dict,
    messages_to_dict,
)

from carag.core.constants import get_constants

constants = get_constants()

_KEY_PREFIX = "carag:chat_history"
_ACTIVITY_KEY_PREFIX = "carag:chat_activity"
_TITLE_KEY_PREFIX = "carag:chat_title"
_OWNER_SEPARATOR = ":"


def _key(session_id: str, owner_id: str) -> str:
    """Return an owner-scoped Redis key for conversation history."""
    return (
        f"{_KEY_PREFIX}"
        f"{_OWNER_SEPARATOR}{owner_id}"
        f"{_OWNER_SEPARATOR}{session_id}"
    )


def _activity_key(owner_id: str) -> str:
    """Return the sorted-set key storing session activity timestamps."""
    return (
        f"{_ACTIVITY_KEY_PREFIX}"
        f"{_OWNER_SEPARATOR}{owner_id}"
    )


def _title_key(session_id: str, owner_id: str) -> str:
    """Return an owner-scoped Redis key for the conversation title."""
    return (
        f"{_TITLE_KEY_PREFIX}"
        f"{_OWNER_SEPARATOR}{owner_id}"
        f"{_OWNER_SEPARATOR}{session_id}"
    )


def get(session_id: str, owner_id: str,) -> list[BaseMessage]:
    """Fetch a conversation from Redis. Returns empty list if missing."""
    data: bytes | None = constants.redis_client.get(
        _key(session_id, owner_id)
    )

    if data is None:
        return []

    # messages_to_dict / from_dict keeps the correct message types
    return messages_from_dict(json.loads(data.decode()))


def set(
    session_id: str,
    owner_id: str,
    messages: list[BaseMessage],
    ex: int | None = None,
) -> None:
    """Store a conversation and update its last-activity timestamp."""

    constants.redis_client.set(
        _key(session_id, owner_id),
        json.dumps(messages_to_dict(messages)).encode(),
        ex=ex,
    )

    # Redis sorted-set score = Unix timestamp.
    constants.redis_client.zadd(
        _activity_key(owner_id),
        {session_id: time.time()},
    )


def get_title(
    session_id: str,
    owner_id: str
) -> str | None:
    """Fetch a stored conversation title."""
    data: bytes | None = constants.redis_client.get(
        _title_key(session_id, owner_id)
    )

    if data is None:
        return None

    if isinstance(data, bytes):
        return data.decode()

    return str(data)


def set_title(
    session_id: str,
    owner_id: str,
    title: str,
) -> None:
    """Store a conversation title."""
    constants.redis_client.set(
        _title_key(session_id, owner_id),
        title,
    )


def truncate(
    session_id: str,
    owner_id: str,
    message_count: int,
) -> list[BaseMessage]:
    """
    Truncate a conversation to the first `message_count` messages.

    Returns the resulting conversation history.
    """

    messages = get(session_id, owner_id)

    if message_count < 0:
        raise ValueError("message_count cannot be negative")

    if message_count > len(messages):
        raise ValueError(
            "message_count cannot exceed the current conversation length"
        )

    truncated_messages = messages[:message_count]

    set(
        session_id,
        owner_id,
        truncated_messages,
    )

    return truncated_messages


def delete(session_id: str, owner_id: str) -> int:
    """Delete a session, its title, and its activity metadata."""

    deleted = constants.redis_client.delete(
        _key(session_id, owner_id)
    )
    constants.redis_client.delete(
        _title_key(session_id, owner_id)
    )
    constants.redis_client.zrem(
        _activity_key(owner_id),
        session_id,
    )

    return deleted


def list_sessions(owner_id: str) -> list[str]:
    """
    Return session IDs ordered from most recently active
    to least recently active.
    """

    return [
        session_id.decode()
        if isinstance(session_id, bytes)
        else session_id
        for session_id in constants.redis_client.zrevrange(
            _activity_key(owner_id),
            0,
            -1,
        )
    ]


def get_last_updated(
    session_id: str,
    owner_id: str,
) -> float | None:
    """Return the last-activity Unix timestamp for a session."""

    timestamp = constants.redis_client.zscore(
        _activity_key(owner_id),
        session_id,
    )

    if timestamp is None:
        return None

    return float(timestamp)


class _HistoryFacade:
    def get(
        self,
        composite_key: str,
        default: Any | None = None,
    ) -> Any:
        """
        Mimics dict.get() so existing code like

            messages_history = conversation_history.get(
                session_id,
                [],
            )

        keeps working.
        """

        if composite_key in self:
            return self[composite_key]

        return default

    def __getitem__(
        self,
        composite_key: str,
    ) -> list[BaseMessage]:
        # composite_key is f"{session_id}||{owner_id}"
        session_id, owner_id = composite_key.split("||", 1)

        return get(session_id, owner_id)

    def __setitem__(
        self,
        composite_key: str,
        messages: list[BaseMessage],
    ) -> None:
        session_id, owner_id = composite_key.split("||", 1)

        set(
            session_id,
            owner_id,
            messages,
        )

    def truncate(
        self,
        session_id: str,
        owner_id: str,
        message_count: int,
    ) -> list[BaseMessage]:
        return truncate(
            session_id,
            owner_id,
            message_count,
        )

    def __delitem__(
        self,
        composite_key: str,
    ) -> None:
        session_id, owner_id = composite_key.split("||", 1)

        delete(
            session_id,
            owner_id,
        )

    def list_sessions(
        self,
        owner_id: str,
    ) -> list[str]:
        return list_sessions(owner_id)

    def get_last_updated(
        self,
        session_id: str,
        owner_id: str,
    ) -> float | None:
        return get_last_updated(
            session_id,
            owner_id,
        )

    def get_title(
        self,
        session_id: str,
        owner_id: str,
    ) -> str | None:
        return get_title(
            session_id,
            owner_id,
        )

    def set_title(
        self,
        session_id: str,
        owner_id: str,
        title: str,
    ) -> None:
        set_title(
            session_id,
            owner_id,
            title,
        )

    def __contains__(
        self,
        composite_key: str,
    ) -> bool:
        session_id, owner_id = composite_key.split("||", 1)

        return (
            constants.redis_client.exists(
                _key(session_id, owner_id)
            )
            == 1
        )


conversation_history: _HistoryFacade = _HistoryFacade()
