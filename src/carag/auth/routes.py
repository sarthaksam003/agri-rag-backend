from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi_users.authentication.strategy.base import Strategy
from fastapi_users.manager import BaseUserManager
from fastapi_users.router.oauth import (
    OAuth2AuthorizeCallback,
    OAuth2Token,
    STATE_TOKEN_AUDIENCE,
    decode_jwt,
    generate_state_token,
)
from fastapi_users.router.oauth import OAuth2AuthorizeResponse
from fastapi_users.exceptions import UserAlreadyExists

from carag.auth.authentication import auth_backend
from carag.auth.manager import fastapi_users, get_user_manager
from carag.auth.models import User, UserRead
from carag.auth.oauth import google_oauth_client
from carag.core.config import get_settings


settings = get_settings()

router = APIRouter()

current_active_user = fastapi_users.current_user(active=True)


@router.get("/me", response_model=UserRead)
async def get_current_user(
    user: User = Depends(current_active_user),
) -> User:
    return user


oauth_callback = OAuth2AuthorizeCallback(
    google_oauth_client,
    redirect_url=f"{settings.BACKEND_URL}/api/auth/google/callback",
)


@router.get(
    "/google/authorize",
    response_model=OAuth2AuthorizeResponse,
)
async def google_authorize(
    request: Request,
    scopes: list[str] = Query(None),
) -> OAuth2AuthorizeResponse:
    redirect_url = f"{settings.BACKEND_URL}/api/auth/google/callback"

    state_data: dict[str, str] = {}
    state = generate_state_token(
        state_data,
        settings.OAUTH_STATE_SECRET,
    )

    authorization_url = await google_oauth_client.get_authorization_url(
        redirect_url,
        state,
        scopes,
    )

    return OAuth2AuthorizeResponse(
        authorization_url=authorization_url,
    )


@router.get("/google/callback")
async def google_callback(
    request: Request,
    access_token_state: tuple[OAuth2Token, str] = Depends(oauth_callback),
    user_manager: BaseUserManager = Depends(get_user_manager),
    strategy: Strategy = Depends(auth_backend.get_strategy),
):
    token, state = access_token_state

    try:
        account_id, account_email = await google_oauth_client.get_id_email(
            token["access_token"]
        )
    except Exception as e:
        print("\n=== GOOGLE PROFILE ERROR ===")
        print("Exception type:", type(e).__name__)
        print("Exception:", e)

        response = getattr(e, "response", None)

        if response is not None:
            print("Status:", response.status_code)
            print("Body:", response.text)

        print("============================\n")
        raise

    if account_email is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OAuth provider did not provide an email address.",
        )

    try:
        decode_jwt(
            state,
            settings.OAUTH_STATE_SECRET,
            [STATE_TOKEN_AUDIENCE],
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid OAuth state.",
        )

    try:
        user = await user_manager.oauth_callback(
            google_oauth_client.name,
            token["access_token"],
            account_id,
            account_email,
            token.get("expires_at"),
            token.get("refresh_token"),
            request,
            associate_by_email=True,
            is_verified_by_default=True,
        )
    except UserAlreadyExists:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email already exists.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User account is inactive.",
        )

    # FastAPI Users creates the Redis session and sets the HTTP-only cookie.
    response = await auth_backend.login(strategy, user)

    await user_manager.on_after_login(
        user,
        request,
        response,
    )

    # Preserve the authentication cookie while redirecting to React.
    response.status_code = status.HTTP_302_FOUND
    response.headers["location"] = f"{settings.FRONTEND_URL}/chat"

    return response


@router.post("/logout", status_code=204)
async def logout(
    request: Request,
    user: User = Depends(current_active_user),
):
    token = request.cookies.get(auth_backend.transport.cookie_name)

    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated.",
        )

    strategy = auth_backend.get_strategy()

    return await auth_backend.logout(
        strategy,
        user,
        token,
    )
