from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from carag.api.routes import chat, documents, sessions
from carag.api.routes.multiquery_chat import router as multiquery_router
from carag.api.routes.tts import router as tts_router
from carag.api.routes.voice_chat import router as voice_router
from carag.auth.database import close_auth_database, init_auth_database
from carag.auth.routes import router as auth_router

from carag.core.config import get_settings


settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_auth_database()

    yield

    close_auth_database()

app = FastAPI(
    title="Carag RAG API",
    description="Multi-tenant RAG Chatbot API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(
    auth_router,
    prefix="/api/auth",
    tags=["auth"],
)
# Include routers
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(
    documents.router, prefix="/api/v1/documents", tags=["documents"])
app.include_router(
    sessions.router, prefix="/api/v1/sessions", tags=["sessions"])
app.include_router(
    multiquery_router, prefix="/api/v1/multiquery-chat", tags=["multiquery-chat"]
)
app.include_router(voice_router, prefix="/api/v1/voice-chat",
                   tags=["voice-chat"])
app.include_router(tts_router, prefix="/api/v1/tts", tags=["tts"])


@app.get("/")
async def root():
    return


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
