## Inzwa Architecture: Real‑Time Shona Speech‑to‑Speech Assistant (Python‑Only)

### Vision and Goals

- Build a state‑of‑the‑art, production‑grade Shona voice assistant that feels instantaneous (<1s end‑to‑end), culturally accurate, safe, and extensible to other African languages.
- 100% Python runtime for all services; no Go components. Favor portable, open‑source stacks with zero/low‑cost deployment paths (e.g., Hugging Face Spaces, Google Colab for training).
- Treat latency, safety, reliability, and privacy as first‑class product features.

### Core Principles

- Streaming by default: incremental ASR, incremental LLM decoding, streaming TTS audio.
- Low‑latency first: quantization, GPU when available, CPU‑only graceful degradation.
- Modularity: swappable ASR/LLM/TTS engines behind stable interfaces; integrate with langpacks for multilingual extension.
- Python everywhere: FastAPI + aiortc/Starlette WebSockets + asyncio orchestration.
- Operational excellence: observability, automated tests, reproducible builds, model versioning.
- Data responsibility: PII‑aware logging, explicit retention, opt‑in data collection.

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
  - FastAPI app exposing:
    - WebRTC (via aiortc) or WebSocket endpoint for streaming audio in/out
    - REST endpoints for batch `asr`, `chat`, `tts`, health, and model info
  - JWT/API key auth, rate limiting, CORS, request validation

- ASR Service (Whisper family)
  - Primary: `faster-whisper` (CTranslate2) for CPU/GPU, streaming, INT8 quantization
  - Alternative: `whisper.cpp` via bindings for CPU‑optimized inference
  - VAD: Silero VAD or WebRTC VAD + optional RNNoise denoise
  - Emits incremental transcripts, timestamps, confidence

- Orchestrator
  - Async Python controller that receives partial transcripts, manages dialogue state, and streams tokens to TTS as they are generated
  - Applies system prompts, tools, safety filters, and routing logic (e.g., fallback to translation, retrieval)
  - Backpressure control and pacing between ASR → LLM → TTS

- LLM Service
  - Base: Mistral‑2B/Gemma‑2B or Phi-3 mini; LoRA fine‑tuned for Shona conversational style
  - Inference: vLLM (GPU) for streaming, KV‑cache reuse, and speculative decoding; fallback: llama‑cpp‑python (CPU, quantized Q4/Q5)
  - Tooling: PEFT/LoRA for fine‑tuning; prompt/response safety filters; optional RAG for knowledge retrieval (e.g., via FAISS embeddings)

- TTS Service
  - Coqui TTS VITS‑lite fine‑tuned on a single Shona voice; ONNX/TorchScript for optimized inference
  - Streams Opus or PCM16 chunks immediately on partial text

- Shared Services
  - Feature flags & configuration (Hydra/pydantic‑settings)
  - Observability (Prometheus metrics, OpenTelemetry traces, structured logs)
  - Model registry & artifact store (Hugging Face Hub / MLflow / local registry)

### End‑to‑End Flow (Streaming)

1) User connects (WebRTC or WS). Mic PCM16 frames sent at 16 kHz.
2) Gateway forwards audio frames → ASR.
3) ASR performs VAD + streaming decode → partial transcripts with timestamps.
4) Orchestrator consumes transcripts, updates conversation state, sends chunks to LLM.
5) LLM streams tokens; Orchestrator buffers into phrase‑sized segments.
6) TTS begins synthesis on earliest phrase, streams audio back to user.
7) Loop continues until end of user turn; barge‑in handled by VAD and session policy.

### Low‑Latency Strategy (Design Targets)

- ASR: faster‑whisper `small`/`base` INT8; aim 200–400 ms per second of audio on CPU; <150 ms on GPU.
- LLM: 2B parameter model, 4‑bit quant on CPU; GPU with vLLM for <200 ms TTFB and 50–200 tok/s.
- TTS: VITS‑lite with ONNX; <200 ms startup, continuous frames at <40 ms chunk cadence.
- Pipeline: start‑speaking time (TTFW) <500 ms; total response <1 s for short utterances.

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

### Reference Repository Structure (Python‑only)

```
src/
  inzwa/
    __init__.py
    api/
      app.py
    asr/
      __init__.py
    config/
      __init__.py
    llm/
      __init__.py
    models/
      __init__.py
    orchestration/
      __init__.py
    telemetry/
      __init__.py
    tts/
      __init__.py
    utils/
      __init__.py
tests/
  test_api.py
scripts/
  train_llm.py
pyproject.toml
Dockerfile
Makefile
docs/
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


