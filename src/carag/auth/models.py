from beanie import Document, PydanticObjectId
from fastapi_users.db import BaseOAuthAccount, BeanieBaseUser
from pydantic import Field
from fastapi_users.schemas import BaseUser


class OAuthAccount(BaseOAuthAccount):
    """OAuth provider account linked to a user."""

    pass


class User(BeanieBaseUser, Document):
    """Application user stored in MongoDB."""

    oauth_accounts: list[OAuthAccount] = Field(default_factory=list)

    class Settings(BeanieBaseUser.Settings):
        name = "users"


class UserRead(BaseUser[PydanticObjectId]):
    """Public representation of an authenticated user."""
