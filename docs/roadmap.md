## Roadmap and Milestones

### Phase 0: Foundations (Week 0, zero cost)

- Repo scaffolding, CI (GH Actions free), container (local Docker), telemetry skeleton, health endpoints

### Phase 1: MVP (<2 s) Weeks 1–8

- W1–2: Data collection bootstrapping (ASR/TTS/LLM) via free sources/Common Voice; initial small models from HF Hub
- W3: Streaming ASR (faster‑whisper), VAD, partials over WS
- W4: LLM LoRA fine‑tune (2B) on Colab; llama‑cpp CPU inference; streaming tokens
- W5–6: Coqui TTS VITS‑lite baseline on Kaggle; streaming synthesis
- W7: Orchestrator streaming pipeline; end‑to‑end working
- W8: Demo deploy (HF Spaces/Fly.io free CPU)

Gate: End‑to‑end P95 ≤ 2 s; intelligibility ≥ 95%; harmlessness pass ≥ 99%

### Phase 2: Latency Push (<1 s) Weeks 9–16

- GPU inference path (vLLM, ONNX)
- Quantization across components; KV‑cache reuse
- Barge‑in and interruption; pacing improvements
- Observability dashboards; load testing to 50 concurrent sessions

Gate: P50 TTFW ≤ 500 ms; P95 E2E ≤ 1.0–1.2 s

### Phase 3: Quality & Scale Weeks 17–24

- Dataset expansion; accent coverage; TTS prosody tuning
- Canary + rollback; A/B for engines; multi‑voice support
- Optional retrieval tools; user opt‑in feedback loops

Gate: MOS ≥ 3.8; WER/CER improved by 20% vs MVP; stable at 100 concurrent sessions

### Phase 4: Community and Expansion (Post-24 weeks)
- Crowdsourcing: Open calls for Shona data contributions (e.g., via GitHub issues, Discord).
- Multilingual: Integrate langpacks for Ndebele/Zulu; community-driven fine-tunes.


