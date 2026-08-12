# Engineering Handoff Manual — `carag` (TTS & Voice Chat)

---

## 1. Architecture Overview

`carag` is a **multi-tenant, GPU-backed RAG (Retrieval-Augmented Generation) API** built on FastAPI + LangGraph. It ingests PDFs, stores chunks as hybrid (dense + sparse) vectors in Qdrant, and answers questions via text or voice pipelines. In this version, the architecture includes Bhasini integration for ASR, Translation, and TTS, allowing full voice-in, translated-RAG, voice-out workflows.

```text
┌──────────────────────────────────────────────────────────────┐
│                        Client / Frontend                      │
└────────────────────────┬─────────────────────────────────────┘
                         │ HTTP  X-Tenant-ID header
                         ▼
┌──────────────────────────────────────────────────────────────┐
│              FastAPI  (src/carag/api/)                        │
│  /api/v1/chat          – Simple RAG text chat                │
│  /api/v1/multiquery-chat – Multi-query RAG text chat         │
│  /api/v1/voice-chat    – Audio in → ASR → RAG → Audio out    │
│  /api/v1/tts           – Text to Speech (Bhasini)            │
│  /api/v1/documents     – Ingest / list / delete PDFs         │
│  /api/v1/sessions      – Chat history CRUD                   │
└──────┬──────────────────────┬────────────────────────────────┘
       │ Celery task           │ direct call
       ▼                       ▼
┌──────────────┐   ┌───────────────────────────────────────────┐
│ Redis (DB 2) │   │   services/                               │
│  task queue  │   │   ├── chat_service.py (Facade for graphs) │
└──────────────┘   │   ├── bhasini.py (ASR/TTS/Translate)      │
                   │   ├── document_ingestion/                 │
┌──────────────┐   │   │     PDF→MD (MineRU vLLM server)       │
│ Redis (DB 1) │   │   │     Chunker (chonkie)                 │
│ chat history │   │   │     Embed → Qdrant                    │
└──────────────┘   │   └── rag_engine/                         │
                   │         simple_rag / mqrag (LangGraph)    │
┌──────────────┐   │         reranker (vLLM)                   │
│   Qdrant     │◄──┘         hallucination_check & guard       │
│ vector store │                                                │
└──────────────┘
```

**Tenancy model:** Every Qdrant point carries `metadata.tenant_id`. All queries filter on this field. `X-Tenant-ID` HTTP header is the sole auth mechanism currently.

---

## 2. Environment & Setup

### 2.1 Python dependencies (uv)

```bash
pip install uv
uv sync                 # installs from uv.lock into .venv
source .venv/bin/activate
```

### 2.2 What uv does NOT cover

| Requirement | Notes |
|---|---|
| **NVIDIA GPU + CUDA driver** | All vLLM containers require CUDA-capable GPU. GPU 0 is assumed. |
| **Docker ≥ 24 + NVIDIA Container Toolkit** | Container orchestration for inference components. |
| **Bhasini API Keys** | External dependency for ASR, Translation, and TTS. |
| **Redis & MongoDB** | Required for chat history and Celery task queue. MongoDB for fastAPI-users auth. |

---

## 3. Codebase Map

```text
src/carag/
├── main.py                      # Entry point: calls uvicorn on api.main.app
├── core/
│   └── ...                      # Configs & Singletons
├── api/
│   ├── main.py                  # FastAPI app; mounts all routers
│   ├── dependencies.py          # get_tenant_id() — reads X-Tenant-ID header
│   ├── history.py               # _HistoryFacade: dict-like wrapper over Redis
│   ├── schemas.py               # All Pydantic request/response models
│   └── routes/
│       ├── chat.py              # POST /api/v1/chat/message (simple RAG)
│       ├── multiquery_chat.py   # POST /api/v1/multiquery-chat/message
│       ├── voice_chat.py        # POST /api/v1/voice-chat/message (ASR->Translate->RAG->Translate)
│       ├── tts.py               # POST /api/v1/tts/synthesize
│       ├── documents.py         # Celery dispatch for ingestion
│       └── sessions.py          # Chat session history CRUD
├── services/
│   ├── chat_service.py          # Central runner for run_rag_pipeline()
│   ├── bhasini.py               # External Bhasini integrations
│   ├── document_ingestion/      # (Excluded in focus) Logic for PDF->MD->Qdrant
│   └── rag_engine/              # (Excluded in focus) LangGraph logic
└── worker/
    └── ...                      # Celery definitions
```

---

## 4. Core Modules — Logic & Pseudocode

### 4.1 `api/routes/voice_chat.py` — Voice to RAG Pipeline
Handles full audio-to-text-to-audio flows by combining Bhasini APIs and RAG.

```python
@router.post("/message"):
    # 1. Transcribe audio (ASR) via Bhasini
    user_text = await transcribe(audio_bytes, source_language)

    # 2. Translate to RAG language if necessary
    rag_input = user_text
    if needs_translation:
        rag_input = await translate(user_text, source_language, target_language)
    
    # 3. RAG Pipeline
    result = await run_rag_pipeline(rag_input, session_id, tenant_id, mode)

    # 4. Translate response back if necessary
    translated_response = None
    if needs_translation:
        translated_response = await translate(result.response, target_language, source_language)

    return VoiceChatResponse(...)
```
**Risk:** Chaining ASR -> Translate -> RAG -> Translate adds significant latency. Failures in any external Bhasini API will break the pipeline without fallback.

### 4.2 `services/chat_service.py` — `run_rag_pipeline`
A facade that abstracts away the specific LangGraph implementation (Simple vs Multiquery).

```python
async def run_rag_pipeline(message, session_id, tenant_id, mode):
    history_key = f"{session_id}||{tenant_id}"
    messages_history = conversation_history.get(history_key, [])
    messages_history.append(HumanMessage(content=message))

    graph = _get_graph(mode) # Lazy load Simple or Multiquery graph
    rag_result = graph.invoke(...)

    ai_response = rag_result["answer"]
    messages_history.append(AIMessage(content=ai_response))
    
    # Trim to last 20 messages to save context window
    conversation_history[history_key] = messages_history[-20:]

    if mode == RagMode.SIMPLE:
        chunks, extra = _extract_simple_result(rag_result)
    else:
        chunks, extra = _extract_multiquery_result(rag_result)

    return RagResult(...)
```

### 4.3 `api/routes/tts.py` — Lazy TTS Endpoint
Provides on-demand Text-to-Speech via Bhasini.

```python
@router.post("/synthesize"):
    audio_bytes = await synthesize(
        text=request.text, 
        target_language=request.language, 
        gender=request.gender
    )
    return Response(content=audio_bytes, media_type="audio/wav")
```
**Notes:** Decoupling TTS from the RAG response generation improves perceived latency; the UI plays audio only when the user requests it.

### 4.4 `api/routes/chat.py` & `multiquery_chat.py`
Exposes the RAG capabilities as standard REST endpoints using the `chat_service.py` facade.

```python
@router.post("/message"):
    result = await run_rag_pipeline(
        message=request.message,
        session_id=request.session_id,
        tenant_id=tenant_id
    )
    return ChatMessageResponse(...)
```

### 4.5 `api/history.py` — `_HistoryFacade`
Handles Redis-backed chat history. Uses composite keys like `carag:chat_history:<tenant_id>:<session_id>`. Provides a dict-like interface over Redis.

```python
class _HistoryFacade:
    def __getitem__(self, composite_key):
        session, tenant = composite_key.split("||")
        data = redis_client.get(key(session, tenant))
        return messages_from_dict(json.loads(data)) if data else []
    
    def __setitem__(self, composite_key, messages):
        session, tenant = composite_key.split("||")
        redis_client.set(key, json.dumps(messages_to_dict(messages)))
```

---

## 5. Data & Models

- **External APIs (Bhasini):** Used for translation (`bhasini.translate`), speech-to-text (`bhasini.transcribe`), and text-to-speech (`bhasini.synthesize`).
- **Redis DB 1:** Chat History. Format: `carag:chat_history:<tenant_id>:<session_id>` -> serialized LangChain messages.
- **Qdrant Vector Store:** Documents stored as dense (1024d) and sparse (BM25) hybrid vectors. Partitioned by `tenant_id`.

[ASK: Who owns the Bhasini API limits and billing credentials for this environment?]

---

## 6. How to Run — Key Workflows

### Start Services
```bash
# Start backend containers (inference, ingestion, observability)
docker compose --profile inference --profile ingestion up -d
source .venv/bin/activate

# Run FastAPI and Celery
carag  
celery -A carag.worker worker --loglevel=info -Q celery
```

### Voice Chat End-to-End
```bash
curl -X POST http://localhost:8000/api/v1/voice-chat/message \
  -H "X-Tenant-ID: my-tenant" \
  -F "audio=@/tmp/demo_voice.wav" \
  -F "source_language=hi" \
  -F "rag_mode=simple"
```

### Lazy TTS
```bash
curl -s -X POST http://localhost:8000/api/v1/tts/synthesize \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: my-tenant" \
  -d '{"text": "Hello world", "language": "en"}' \
  --output /tmp/speech.wav
```

---

## 7. Known Issues, Bugs, Workarounds

1. **`get_tenant_id` caching:** `dependencies.py` applies `@lru_cache` to `get_tenant_id`. This creates a global process cache of the first `X-Tenant-ID` header processed, breaking multi-tenancy.
2. **Synchronous Celery Embeddings:** Ingestion celery tasks (`worker/tasks.py`) block synchronously on embedding model generation, causing queue pile-ups.
3. **External API dependency (Bhasini):** If Bhasini goes down, the Voice Chat route will crash without a fallback text mode.
4. **No exception handling in RAG execution:** Underlying errors in LangGraph execution bubble up as HTTP 500 without graceful degradation.
5. **Session Truncation:** `run_rag_pipeline` hard-truncates the session to 20 messages, losing earlier conversation context silently.

---

## 8. Testing & Validation

- Run unit and integration tests using `pytest tests/ -v`.
- Requires mocked or live Bhasini keys in `.env`.
- Ensure all docker-compose profiles are running for end-to-end integration tests.

---

## 9. Deployment / Operational Notes

- **GPU Memory Constraints:** If `vllm-llm`, `vllm-embed`, `vllm-rerank`, and `mineru-vllm-server` all share a single GPU (Device 0), memory limits will be very tight (requires >=24GB VRAM). 
- **Redis State:** Chat history is maintained in Redis DB 1. Use the `api/v1/sessions` endpoint to delete specific sessions to prevent Redis OOM over time.
- **Observability:** Ensure Langfuse environment variables are populated correctly in `.env` to track Voice Chat latency (ASR -> RAG -> TTS).
