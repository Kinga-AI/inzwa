## Technical Specifications: Inzwa (Python‑Only)

### Runtime and Dependencies (Per .cursorrules)

- Python ≥ 3.11
- **Ultra-light stack only**:
  - FastAPI with ORJSONResponse, uvicorn[standard]
  - WebSocket via Starlette (default), aiortc/WebRTC (flag only)
  - pydantic v2 (BaseModel, BaseSettings)
  - httpx.AsyncClient (single DI instance, timeouts, retries)
- **ASR**: faster-whisper (CTranslate2) INT8, whisper.cpp (CPU)
- **LLM**: llama-cpp-python Q4/Q5 (CPU-first), vLLM (GPU flag only)
- **TTS**: Coqui TTS VITS-lite (Torch/ONNX), streaming
- **Metrics**: prometheus-client only (no OpenTelemetry unless needed)
- **Optional**: Redis (cache/ratelimit), Postgres (only if truly needed)

### Media and Codecs

- Input audio: PCM16 LE, mono, 16 kHz (WebSocket) or Opus 48 kHz (WebRTC)
- Output audio: Opus 48 kHz preferred; PCM16 16 kHz fallback
- Chunk duration: 20–40 ms; MTU‑aware framing

### ASR Subsystem

- Default engine: faster‑whisper `small` or `base` (INT8)
- Streaming: sliding window with VAD gating; emit `TranscriptChunk`
- Hotwords/biasing: word‑level bias lists (if supported) for Shona terms
- Latency targets: 0.2–0.5 s for typical utterances
- Resource: CPU‑only mode RAM ~1–2 GB; GPU VRAM ~2–4 GB small models

### LLM Subsystem

- Base model: Mistral-2B with Shona LoRA
- **CPU-first**: llama-cpp-python Q4_K_M default
  - Context: 4k tokens max
  - Performance: ≥10-30 tok/s on modern CPU
  - Reduce max_new_tokens under load
- **GPU optional**: vLLM (flag only)
  - TTFB: ≤200-400ms
  - Throughput: ≥50-200 tok/s
- Safety: Simple regex filters only (no heavy frameworks)

### TTS Subsystem

- Model: Coqui TTS VITS‑lite fine‑tuned on 5–10 h single‑speaker Shona
- Export: ONNX for low‑latency inference; batch size 1, streaming synthesis
- Audio output: Opus bitrate 24–32 kbps; optional WAV for offline
- Latency: <0.5 s sentence; <0.2 s TTFW

### Orchestrator (Ultra-light)

- **<50 lines per function rule**
- Bounded queues: max 8 items (strict)
- Simple phrase boundary: punctuation check
- Minimal pipeline: ASR partials → LLM tokens → phrase chunks → TTS frames
- Backpressure: await on queues, drop low-value partials under stress

### Configuration (Per .cursorrules template)

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INZWA_", case_sensitive=False)
    debug: bool = False
    cors_allowed_origins: str = "http://localhost:7860"
    request_timeout_s: float = 5.0
    max_text_chars: int = 400
    max_audio_seconds: int = 20
    enable_webrtc: bool = False  # Flag only
    asr_engine: str = "faster-whisper"
    asr_model: str = "small"
    llm_engine: str = "llama-cpp"  # CPU-first
    llm_model: str = "mistral-2b-shona-lora"
    tts_engine: str = "coqui-vits-lite"
    audio_out_codec: str = "opus"
```

### API Surface (Tiny per .cursorrules)

```
WS   /ws/audio     : bidirectional audio + JSON events (default transport)
POST /v1/tts       : text → audio stream (Opus preferred)
POST /v1/chat      : messages[] → streaming tokens (SSE/WS)
GET  /healthz      : liveness
GET  /readyz       : models ready, caches warm
POST /v1/admin/warmup : load models, return versions + checksums
```

**Note**: /v1/asr endpoint removed per .cursorrules (use WebSocket for ASR)

### WebSocket Message Types (excerpt)

```json
// Client -> Server (control)
{"type": "start", "session_id": "...", "auth": "...", "codec": "pcm16", "sample_rate": 16000}
{"type": "end_turn"}

// Server -> Client (events)
{"type": "asr.partial", "text": "Mhoro", "start_ms": 0, "end_ms": 300}
{"type": "llm.partial", "token": "Mh"}
{"type": "tts.start"}
```

### Model and Artifact Management

- Storage: Hugging Face Hub for weights; optional local cache; MLflow registry for versions
- Integrity: SHA256 checks; signed releases; SBOM for containers
- Warmup: `POST /v1/admin/warmup` loads models and runs 1 dummy request

### Performance Budgets

- P50 TTFW (first TTS audio): ≤ 500 ms
- P95 total round trip (short utterance): ≤ 1200 ms
- Server CPU usage at 10 concurrent sessions: ≤ 80% per core target

### Compatibility and Fallbacks

- If WebRTC unsupported, use WebSocket PCM16 path
- If GPU unavailable, switch to CPU engines and reduce model sizes


