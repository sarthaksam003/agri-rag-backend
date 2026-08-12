from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

from carag.core.constants import get_constants

constants = get_constants()

_KEY_PREFIX = "carag:chat_history"
_TENANT_SEPARATOR = ":"


def _key(session_id: str, tenant_id: str) -> str:
    """Return a tenant-scoped Redis key: carag:chat_history:<tenant>:<session>"""
    return f"{_KEY_PREFIX}{_TENANT_SEPARATOR}{tenant_id}{_TENANT_SEPARATOR}{session_id}"


def get(session_id: str, tenant_id: str) -> list[BaseMessage]:
    """Fetch a conversation from Redis.  Returns empty list if missing."""
    data: bytes | None = constants.redis_client.get(_key(session_id, tenant_id))
    if data is None:
        return []
    # messages_to_dict / from_dict keeps the correct message types
    return messages_from_dict(json.loads(data.decode()))


def set(
    session_id: str,
    tenant_id: str,
    messages: list[BaseMessage],
    ex: int | None = None,
) -> None:
    """Store (or overwrite) a conversation.  Optional TTL in seconds."""
    constants.redis_client.set(
        _key(session_id, tenant_id),
        json.dumps(messages_to_dict(messages)).encode(),
        ex=ex,
    )


def delete(session_id: str, tenant_id: str) -> int:
    """Delete a session.  Returns the number of keys removed (0 or 1)."""
    return constants.redis_client.delete(_key(session_id, tenant_id))


def list_sessions(tenant_id: str) -> list[str]:
    """Return all session-IDs for a tenant (without the prefix)."""
    pattern = f"{_KEY_PREFIX}{_TENANT_SEPARATOR}{tenant_id}{_TENANT_SEPARATOR}*"
    keys = constants.redis_client.scan_iter(match=pattern, count=500)
    # strip prefix + tenant
    prefix_len = len(_KEY_PREFIX) + len(tenant_id) + 2
    return [k.decode()[prefix_len:] for k in keys]


class _HistoryFacade:
    def get(self, composite_key: str, default: Any | None = None) -> Any:
        """
        Mimics dict.get() so existing code like
            messages_history = conversation_history.get(session_id, [])
        keeps working.
        """
        if composite_key in self:
            return self[composite_key]
        return default

    def __getitem__(self, composite_key: str) -> list[BaseMessage]:
        # composite_key is f"{session_id}||{tenant_id}" (or change to taste)
        session_id, tenant_id = composite_key.split("||", 1)
        return get(session_id, tenant_id)

    def __setitem__(self, composite_key: str, messages: list[BaseMessage]) -> None:
        session_id, tenant_id = composite_key.split("||", 1)
        set(session_id, tenant_id, messages)

    def __delitem__(self, composite_key: str) -> None:
        session_id, tenant_id = composite_key.split("||", 1)
        delete(session_id, tenant_id)

    def __contains__(self, composite_key: str) -> bool:
        session_id, tenant_id = composite_key.split("||", 1)
        return constants.redis_client.exists(_key(session_id, tenant_id)) == 1

    def keys(self):
        """Iterate over *all* composite keys in DB-1 (use sparingly)."""
        pattern = f"{_KEY_PREFIX}{_TENANT_SEPARATOR}*"
        for key in constants.redis_client.scan_iter(match=pattern, count=500):
            # key is b'carag:chat_history:<tenant>:<session>'
            _, _, tenant, session = key.decode().split(_TENANT_SEPARATOR, 3)
            yield f"{session}||{tenant}"


conversation_history: _HistoryFacade = _HistoryFacade()
