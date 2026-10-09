from beanie import Document, PydanticObjectId
from fastapi_users.db import BaseOAuthAccount, BeanieBaseUser
from pydantic import BaseModel, Field
from fastapi_users.schemas import BaseUser


class OAuthAccount(BaseOAuthAccount):
    """OAuth provider account linked to a user."""

    pass


class User(BeanieBaseUser, Document):
    """Application user stored in MongoDB."""

    oauth_accounts: list[OAuthAccount] = Field(default_factory=list)
    profile_picture: str | None = None

    class Settings(BeanieBaseUser.Settings):
        name = "users"


class UserRead(BaseUser[PydanticObjectId]):
    """Public representation of an authenticated user."""

    profile_picture: str | None = None


class ProfilePictureUpdate(BaseModel):
    """Request body for updating the authenticated user's profile picture."""

    profile_picture: str | None = None
