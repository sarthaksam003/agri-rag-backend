from celery import Celery

from carag.core.config import get_settings

settings = get_settings()

auth_part = f":{settings.REDIS_AUTH}@" if settings.REDIS_AUTH else ""

celery_base_url = f"redis://{auth_part}{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.CELERY_REDIS_DB}"
celery_app = Celery(
    "carag",
    broker=celery_base_url,
    backend=celery_base_url,
    include=["carag.worker.tasks"],
)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)
