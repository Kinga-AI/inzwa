## Technical Specifications: Inzwa (Python‑Only)

### Runtime and Dependencies

- Python ≥ 3.10
- FastAPI, Uvicorn, Starlette
- aiortc (WebRTC) and/or WebSocket (websockets/starlette)
- ASR: faster‑whisper (CTranslate2) or whisper.cpp bindings; silero‑vad; rnnoise (optional)
- LLM: vLLM client (GPU) or llama‑cpp‑python (CPU); transformers/PEFT for LoRA
- TTS: Coqui TTS (VITS‑lite), onnxruntime, torchaudio
- Telemetry: prometheus‑client, opentelemetry‑sdk, structlog/loguru
- Config: pydantic‑settings, hydra‑core (optional)

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

- Base models: Mistral‑2B/Gemma‑2B
- Fine‑tuning: LoRA (rank 8–16), bf16/fp16 on a single 24 GB GPU; 1–3 epochs; cosine LR
- Inference:
  - GPU path: vLLM with tensor parallel=1, paged attention; streaming via SSE/WebSocket
  - CPU path: llama‑cpp‑python Q4_K_M; context 4k tokens; 10–30 tok/s on modern CPU
- Prompting: system prompt tuned for Shona style; tools gated behind keywords
- Safety: pre‑ and post‑generation regex filters; refusal templates

### TTS Subsystem

- Model: Coqui TTS VITS‑lite fine‑tuned on 5–10 h single‑speaker Shona
- Export: ONNX for low‑latency inference; batch size 1, streaming synthesis
- Audio output: Opus bitrate 24–32 kbps; optional WAV for offline
- Latency: <0.5 s sentence; <0.2 s TTFW

### Orchestrator

- Async Python tasks with bounded queues; cancellation & timeouts per stage
- Phrase detection from token stream using punctuation heuristics and timing
- Barge‑in: monitor VAD during TTS; on user speech, pause TTS and prioritize ASR

### Configuration (pydantic‑settings)

```python
class Settings(BaseSettings):
    enable_webrtc: bool = True
    asr_engine: str = "faster-whisper"  # or "whisper.cpp"
    asr_model: str = "small"
    llm_engine: str = "vllm"            # or "llama-cpp"
    llm_model: str = "mistral-2b-shona-lora"
    tts_engine: str = "coqui-vits-lite"
    audio_out_codec: str = "opus"
    max_concurrent_sessions: int = 50
```

### Public REST API Schemas (excerpt)

```json
POST /v1/asr
Request: { "audio_base64": "...", "sample_rate": 16000 }
Response: { "text": "...", "segments": [ { "text": "...", "start_ms": 0, "end_ms": 850 } ] }

POST /v1/chat
Request: { "messages": [ {"role": "user", "content": "Mhoro"} ], "stream": true }
Response (SSE/stream): data: {"token": "M"}\n data: {"token": "h"} ...

POST /v1/tts
Request: { "text": "Mhoro dunia", "voice": "shona_female_a" }
Response: audio/opus stream
```

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


