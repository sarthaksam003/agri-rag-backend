from collections.abc import AsyncGenerator

from beanie import PydanticObjectId
from fastapi import Depends, Request
from fastapi_users import BaseUserManager, FastAPIUsers
from fastapi_users.db import ObjectIDIDMixin

from carag.auth.authentication import auth_backend
from carag.auth.database import get_user_db
from carag.auth.models import User
from carag.core.config import get_settings


settings = get_settings()


class UserManager(ObjectIDIDMixin, BaseUserManager[User, PydanticObjectId]):
    """Manage application users."""

    reset_password_token_secret = settings.AUTH_SECRET
    verification_token_secret = settings.AUTH_SECRET

    async def on_after_register(
        self,
        user: User,
        request: Request | None = None,
    ) -> None:
        return None


async def get_user_manager(
    user_db=Depends(get_user_db),
) -> AsyncGenerator[UserManager, None]:
    yield UserManager(user_db)


fastapi_users = FastAPIUsers[User, PydanticObjectId](
    get_user_manager,
    [auth_backend],
)

current_active_user = fastapi_users.current_user(active=True)