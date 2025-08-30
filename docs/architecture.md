## Izwi Architecture: Real‑Time Shona Speech‑to‑Speech Assistant (Python‑Only)

### Vision and Goals

- Build a state‑of‑the‑art, production‑grade Shona voice assistant that feels instantaneous (<1s end‑to‑end), culturally accurate, safe, and extensible to other African languages.
- 100% Python runtime for all services; no Go components. Favor portable, open‑source stacks with zero/low‑cost deployment paths (e.g., Hugging Face Spaces, Google Colab for training).
- Treat latency, safety, reliability, and privacy as first‑class product features.

### Core Principles (Per .cursorrules)

- **Very low latency** first: streaming everywhere, bounded queues (max 8), small chunks (20–40 ms)
- **Very lightweight**: minimum deps, minimal routes, minimal configuration
- **High security**: tight CORS, TLS, API keys/JWT, rate limits; no raw payload logs
- **Great usability**: one clean UI, "it just works"; sane defaults; clear errors
- **Clean code**: typed, mypy-friendly, small functions (<50 lines), no utils dumping ground
- **Quantize & cache**: Q4/Q5/INT8, warm models, read-through caches, idempotent renders

### High‑Level Architecture

```mermaid
graph TD
  U[User: Mic & Speaker] -- WebRTC/WebSocket --> G[Gateway API]
  subgraph Realtime Pipeline (Python)
    G --> A[ASR Service\nWhisper.cpp / faster-whisper]
    A --> O[Orchestrator\nAsync pipeline + Dialogue State]
    O --> L[LLM Service\nMistral/Gemma 2B LoRA via vLLM or llama.cpp]
    L --> O
    O --> T[TTS Service\nCoqui TTS VITS-lite / ONNX]
  end
  T -- Opus PCM chunks --> G -- Audio --> U

  classDef svc fill:#e7f0ff,stroke:#5682f9,stroke-width:1px
  class A,O,L,T,G svc
```

### Component Overview

- Gateway API
  - FastAPI app with ORJSONResponse (ultra-fast)
  - WebSocket `/ws/audio` for bidirectional streaming (default)
  - Minimal REST endpoints: `/v1/tts`, `/v1/chat` (SSE/WS)
  - `/healthz`, `/readyz`, `/v1/admin/warmup` only
  - JWT/API key with per-key quotas; strict CORS (no wildcards)

- ASR Service (Whisper family)
  - Primary: `faster-whisper` (CTranslate2) for CPU/GPU, streaming, INT8 quantization
  - Alternative: `whisper.cpp` via bindings for CPU‑optimized inference
  - VAD: Silero VAD or WebRTC VAD + optional RNNoise denoise
  - Emits incremental transcripts, timestamps, confidence

- Orchestrator (Ultra-light, <50 lines)
  - Minimal async controller: ASR partials → LLM tokens → phrase chunks → TTS frames
  - Bounded queues (max 8 items per .cursorrules)
  - Simple phrase boundary detection
  - Backpressure by awaiting on bounded queues; drop low-value partials under stress

- LLM Service
  - Base: Mistral-2B with Shona LoRA (ultra-light)
  - Primary: llama-cpp-python Q4_K_M (CPU-first)
  - Optional: vLLM (GPU) as flag only
  - Reduce max_new_tokens under load
  - No heavy tooling; safety via simple regex filters

- TTS Service
  - Coqui TTS VITS‑lite fine‑tuned on a single Shona voice; ONNX/TorchScript for optimized inference
  - Streams Opus or PCM16 chunks immediately on partial text

- Ops Services (Minimal)
  - Settings via pydantic-settings with IZWI_ prefix
  - Prometheus metrics only (no OpenTelemetry unless needed)
  - No raw audio/text logging; HMAC user_hash only
  - Model checksums verified on load

### End‑to‑End Flow (Streaming)

1) User connects (WebRTC or WS). Mic PCM16 frames sent at 16 kHz.
2) Gateway forwards audio frames → ASR.
3) ASR performs VAD + streaming decode → partial transcripts with timestamps.
4) Orchestrator consumes transcripts, updates conversation state, sends chunks to LLM.
5) LLM streams tokens; Orchestrator buffers into phrase‑sized segments.
6) TTS begins synthesis on earliest phrase, streams audio back to user.
7) Loop continues until end of user turn; barge‑in handled by VAD and session policy.

### Performance Budgets (Per .cursorrules)

- P50 TTFW: ≤ 500ms (first audio frame out)
- P95 round-trip: ≤ 1200ms (short utterance)
- ASR RTF: 0.2–0.5× (faster-whisper small/base INT8)
- LLM TTFB: ≤ 600–900ms CPU Q4 | ≤ 200–400ms GPU
- TTS TTFW: 200–300ms
- Token rate: ≥10–30 tok/s CPU | ≥50–200 tok/s GPU

### Protocols and Media

- Ingress audio: Opus over WebRTC (preferred) or raw PCM16 over WebSocket
- Egress audio: Opus (48 kHz) or PCM16 (16 kHz) chunks, configurable
- Chunk sizes: 20–40 ms frames; JSON control messages interleaved on data channel/WS
- Time sync: client timestamping + server NTP drift correction

### Error Handling and Resilience

- Graceful degradation: switch to smaller models on overload; shed load by disabling TTS prosody features
- Supervisors: watchdogs for ASR/LLM/TTS subprocesses; automatic warm‑restart
- Backpressure: bounded async queues; drop lowest‑priority partials when congested
- Idempotent control messages; explicit session state transitions

### Security and Privacy (see dedicated doc)

- TLS everywhere; SRTP/DTLS for WebRTC
- Minimal logging of raw audio/text; hashing and redaction; opt‑in dataset contribution
- Signed model artifacts; SBOM and container image scanning

### Observability

- Metrics: TTFW, TRL, ASR WER/CER (eval), token throughput, synthesis RTF, GPU/CPU utilization
- Tracing: ASR→LLM→TTS spans with correlation IDs per session
- Logs: structured JSON with session and turn IDs; privacy‑aware payloads

### Modularity and Swappability

- Each ML subsystem behind a Python interface:
  - `ASR.transcribe_stream(frames) -> AsyncIterator[TranscriptChunk]`
  - `LLM.generate_stream(messages) -> AsyncIterator[TokenChunk]`
  - `TTS.synthesize_stream(text) -> AsyncIterator[AudioChunk]`
- Enables A/B of engines without touching the orchestrator contract.

### Reference Repository Structure (Per .cursorrules)

```
src/
  izwi/
    api/          # FastAPI app, routes, middleware
    asr/          # ASR adapters
    llm/          # LLM adapters  
    tts/          # TTS adapters
    orch/         # Tiny orchestrator + queues (was orchestration/)
    ops/          # Settings, logging, metrics, limits (was config/ + telemetry/)
    ui/           # Minimal web UI
    models/       # Pydantic schemas
    utils/        # Minimal helpers only
tests/          # Unit + light integration
scripts/
pyproject.toml
Dockerfile
Makefile
docs/
TASKS.md        # MVP task tracking
```

### Configuration and Feature Flags

- pydantic‑settings for env‑driven config; optional Hydra for layered profiles
- Flags: enable_webrtc, model_size, quantization_level, gpu_policy, streaming_tts

### Internationalization and Shona Specifics

- Shona tokenizer/orthography normalization in ASR post‑proc
- Domain lexicon injection (names, places) through ASR biasing (if supported)
- LLM prompt templates tailored to Shona politeness and idioms
- TTS voice tuned for Shona prosody and phonotactics

### Compliance and Licensing

- Apache‑2.0 project; ensure model/data licenses are compatible (Hugging Face model cards)
- Respect site ToS for crawled data; maintain dataset manifests with provenance

### Risks and Mitigations (high‑level)

- Limited Shona data → self‑recording, university partnerships, Common Voice; synthetic augmentation
- Latency on CPU → quantization + streaming; optionally GPU on Colab/Fly.io GPU machines
- TTS quality variance → invest in clean dataset, speaker consistency, noise control

### Acceptance Criteria (Architecture Level)

- Single‑binary (container) dev deployment runs end‑to‑end on CPU‑only laptop
- Streaming round‑trip works with <2 s initial MVP; <1 s after optimization
- Swappable engines demonstrated by toggling ASR (faster‑whisper vs whisper.cpp)

### Edge and Offline Capabilities
For ultra-low latency in zero-budget scenarios:
- Browser-based edge: Use WebAssembly (e.g., whisper.cpp WASM) for client-side ASR/TTS on capable devices, falling back to server.
- Offline mode: Package models for local run (e.g., via ONNX Runtime Web or mobile SDKs like TensorFlow Lite).
- Hybrid: Client handles VAD and initial buffering; server for heavy lifting.


