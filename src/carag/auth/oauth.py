from httpx_oauth.clients.google import GoogleOAuth2

from carag.core.config import get_settings


settings = get_settings()


google_oauth_client = GoogleOAuth2(
    client_id=settings.GOOGLE_CLIENT_ID,
    client_secret=settings.GOOGLE_CLIENT_SECRET.get_secret_value(),
)