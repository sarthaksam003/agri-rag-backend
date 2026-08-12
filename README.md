# Installation

- install uv
- run `uv sync`

# TODO Testing

Test out

- [x] ./src/carag/custom/rerank.py
  - for this an implementation of api interaction can be seen in ./src/carag/core/constants.py
  - put the test file in ./tests/unit
  - use pytest as testing framework

* [x] TODO: add data/1.pdf
  - add data/2.pdf
  - pytest tests/integration/ -v


- [] JWT Token generation and auto tenant id generation 
- [] Test every single thing out. including all the functions sub functions etc.
- [] Add tracing and tracking through Langfuse


START THE DOCKER PROCESS
- `docker compose --profile inference --profile observability up -d && sleep 30 && docker compose --profile ingestion --profile inference --profile observability up -d`
START THE WEBSERVER
- `carag`
START celery worker
- `celery -A carag.worker worker --loglevel=info -Q celery`


# NOHUP COMMANDS
- nohup celery -A carag.worker worker --loglevel=info -Q celery > celery_worker.log 2>&1 &
- nohup python3 src/carag/main.py > main.log 2>&1 &


# TESTING

# ============================================
# 2. TEXT CHAT — Simple RAG
# ============================================
echo "\n=== Text Chat (Simple RAG) ==="
curl -s -X POST http://localhost:38779/api/v1/chat/message \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: YOUR_TENANT" \
  -d '{"message": "What are the key features of this product?", "session_id": "demo-session-1"}' \
  | jq .

# ============================================
# 3. TEXT CHAT — Follow-up (shows conversation memory)
# ============================================
echo "\n=== Follow-up (same session, proves memory) ==="
curl -s -X POST http://localhost:38779/api/v1/chat/message \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: YOUR_TENANT" \
  -d '{"message": "Can you explain the first point in more detail?", "session_id": "demo-session-1"}' \
  | jq .

# ============================================
# 4. MULTIQUERY CHAT — Show query generation
# ============================================
echo "\n=== Multiquery RAG (generates query variations) ==="
curl -s -X POST http://localhost:38779/api/v1/multiquery-chat/message \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: YOUR_TENANT" \
  -d '{"message": "What are the eligibility criteria?"}' \
  | jq .

# ============================================
# 5. TTS — Generate Hindi audio (lazy, on-demand)
# ============================================
echo "\n=== TTS: Generate speech for demo ==="
curl -s -X POST http://localhost:38779/api/v1/tts/synthesize \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: YOUR_TENANT" \
  -d '{"text": "ଏହି ଉତ୍ପାଦର ମୁଖ୍ୟ ବିଶେଷତାଗୁଡ଼ିକ କ’ଣ?", "language": "or"}' \
  --output /tmp/demo_voice.wav
echo "Saved audio to /tmp/demo_voice.wav"

# Play it (optional, if on mac)
# afplay /tmp/demo_voice.wav

# ============================================
# 6. VOICE CHAT — Full pipeline (audio in → text out)
# ============================================
echo "\n=== Voice Chat: Audio → ASR → Translate → RAG → Translate back ==="
curl -s -X POST http://localhost:38779/api/v1/voice-chat/message \
  -H "X-Tenant-ID: YOUR_TENANT" \
  -F "audio=@/tmp/demo_voice.wav" \
  -F "source_language=hi" \
  -F "rag_mode=simple" \
  | jq .

# ============================================
# 7. VOICE CHAT + MULTIQUERY — Same audio, different RAG mode
# ============================================
echo "\n=== Voice Chat with Multiquery RAG ==="
curl -s -X POST http://localhost:38779/api/v1/voice-chat/message \
  -H "X-Tenant-ID: YOUR_TENANT" \
  -F "audio=@/tmp/demo_voice.wav" \
  -F "source_language=hi" \
  -F "rag_mode=multiquery" \
  | jq .

# ============================================
# 8. LAZY TTS — Speak the RAG response (user clicks 🔊)
# ============================================
echo "\n=== Lazy TTS: Speak the translated response ==="
curl -s -X POST http://localhost:38779/api/v1/tts/synthesize \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: YOUR_TENANT" \
  -d '{"text": "प्रदान की गई सामग्री के आधार पर इस प्रश्न का उत्तर देने के लिए मेरे पास पर्याप्त जानकारी नहीं है।", "language": "hi", "gender": "female"}' \
  --output /tmp/demo_response.wav
echo "Saved response audio to /tmp/demo_response.wav"
