import redis.asyncio as redis
from fastapi_users.authentication import (
    AuthenticationBackend,
    CookieTransport,
    RedisStrategy,
)

from carag.core.config import get_settings

settings = get_settings()


# ---------------------------------------------------------------------------
# Redis authentication/session storage
# ---------------------------------------------------------------------------

auth_redis = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    password=settings.REDIS_AUTH,
    db=settings.RED_AUTH_DB,
    decode_responses=True,
)


def get_redis_strategy() -> RedisStrategy:
    return RedisStrategy(
        auth_redis,
        lifetime_seconds=60 * 60 * 24 * 7,
        key_prefix="agri_auth:",
    )


# ---------------------------------------------------------------------------
# HTTP-only cookie transport
# ---------------------------------------------------------------------------

cookie_transport = CookieTransport(
    cookie_name="agri_auth",
    cookie_max_age=60 * 60 * 24 * 7,
    cookie_secure=False,
    cookie_httponly=True,
    cookie_samesite="lax",
)


# ---------------------------------------------------------------------------
# Authentication backend
# ---------------------------------------------------------------------------

auth_backend = AuthenticationBackend(
    name="cookie",
    transport=cookie_transport,
    get_strategy=get_redis_strategy,
)