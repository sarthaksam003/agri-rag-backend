# from uuid import uuid4
#
# from langchain_core.documents import Document
# from langchain_qdrant import QdrantVectorStore
# from qdrant_client import QdrantClient, models

from carag.core.config import get_settings
from carag.core.constants import get_constants

settings = get_settings()
constants = get_constants()

QDRANT_URL = f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}"

client = QdrantClient(url=QDRANT_URL)
