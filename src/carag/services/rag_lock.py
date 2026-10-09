import uuid

import redis.asyncio as redis

from carag.core.config import get_settings


settings = get_settings()


rag_lock_redis = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    password=settings.REDIS_AUTH,
    db=settings.REDIS_CHAT_DB,
    decode_responses=True,
)


LOCK_KEY_PREFIX = "agrichat:rag-lock:"

# The lock remains valid for this long unless the active request
# renews it through the heartbeat.
RAG_LOCK_LEASE_SECONDS = 60

# The active request renews its lease at this interval.
RAG_LOCK_HEARTBEAT_SECONDS = 20


async def acquire_rag_lock(user_id: str) -> str | None:
    """
    Acquire a per-user RAG lock.

    Returns a unique lock token if acquired.
    Returns None if this user already has an active RAG request.
    """
    lock_token = str(uuid.uuid4())
    lock_key = f"{LOCK_KEY_PREFIX}{user_id}"

    acquired = await rag_lock_redis.set(
        lock_key,
        lock_token,
        nx=True,
        ex=RAG_LOCK_LEASE_SECONDS,
    )

    return lock_token if acquired else None


async def renew_rag_lock(user_id: str, lock_token: str) -> bool:
    """
    Renew the user's RAG lock only if the stored token still matches.

    Returns True if the lock was successfully renewed.
    Returns False if the lock no longer belongs to this request.
    """
    lock_key = f"{LOCK_KEY_PREFIX}{user_id}"

    result = await rag_lock_redis.eval(
        """
        if redis.call("GET", KEYS[1]) == ARGV[1] then
            return redis.call("EXPIRE", KEYS[1], ARGV[2])
        end
        return 0
        """,
        1,
        lock_key,
        lock_token,
        RAG_LOCK_LEASE_SECONDS,
    )

    return bool(result)


async def release_rag_lock(user_id: str, lock_token: str) -> None:
    """
    Release the user's RAG lock only if the stored token matches.

    This prevents one request from accidentally deleting another
    request's lock.
    """
    lock_key = f"{LOCK_KEY_PREFIX}{user_id}"

    stored_token = await rag_lock_redis.get(lock_key)

    if stored_token == lock_token:
        await rag_lock_redis.delete(lock_key)