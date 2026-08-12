from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from carag.api.routes import chat, documents, sessions
from carag.api.routes.multiquery_chat import router as multiquery_router
from carag.api.routes.tts import router as tts_router
from carag.api.routes.voice_chat import router as voice_router

app = FastAPI(
    title="Carag RAG API", description="Multi-tenant RAG Chatbot API", version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(documents.router, prefix="/api/v1/documents", tags=["documents"])
app.include_router(sessions.router, prefix="/api/v1/sessions", tags=["sessions"])
app.include_router(
    multiquery_router, prefix="/api/v1/multiquery-chat", tags=["multiquery-chat"]
)
app.include_router(voice_router, prefix="/api/v1/voice-chat", tags=["voice-chat"])
app.include_router(tts_router, prefix="/api/v1/tts", tags=["tts"])


@app.get("/")
async def root():
    return


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
