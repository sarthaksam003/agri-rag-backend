from collections.abc import AsyncGenerator

from beanie import init_beanie
from fastapi_users.db import BeanieUserDatabase
from motor.motor_asyncio import AsyncIOMotorClient

from carag.auth.models import OAuthAccount, User
from carag.core.config import get_settings


settings = get_settings()

mongo_client = AsyncIOMotorClient(
    host=settings.MONGO_AUTH_HOST,
    port=settings.MONGO_AUTH_PORT,
    username=settings.MONGO_AUTH_USERNAME,
    password=settings.MONGO_AUTH_PASSWD,
    authSource="admin",
    authMechanism="SCRAM-SHA-256",
)

database = mongo_client[settings.MONGO_AUTH_DB]


async def init_auth_database() -> None:
    await init_beanie(
        database=database,
        document_models=[User],
    )


def close_auth_database() -> None:
    mongo_client.close()


async def get_user_db() -> AsyncGenerator[BeanieUserDatabase[User], None]:
    yield BeanieUserDatabase(User, OAuthAccount)