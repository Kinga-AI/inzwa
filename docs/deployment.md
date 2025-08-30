## Deployment Guide: Izwi (Python‑Only)

### Targets

- Local dev (CPU‑only, zero cost)
- Hugging Face Spaces (free CPU tier for demos, up to 1GB RAM, persistent storage)
- Fly.io (free tier: 3 machines, 256MB RAM each, 3GB volumes; GPU add-ons from $0.02/hr but start CPU-only)
- Optional: Kubernetes for scale (e.g., free Minikube local or low-cost managed like DigitalOcean Kubernetes ~$10/mo)

### Containerization

- Base: `python:3.10-slim`
- System deps: `ffmpeg`, `libsndfile1`, `libopus0`
- Python deps: FastAPI, uvicorn, aiortc, faster‑whisper, llama‑cpp‑python, TTS, onnxruntime
- Cache model weights at build or first‑run warmup

### Example Compose (excerpt)

```yaml
services:
  izwi:
    build: .
    ports: ["8000:8000"]
    environment:
      - ENABLE_WEBRTC=true
      - ASR_MODEL=small
      - LLM_ENGINE=llama-cpp
      - TTS_ENGINE=coqui
    volumes:
      - ./weights:/app/weights
```

### Runtime

- Run: `uvicorn izwi.api.app:app --host 0.0.0.0 --port 8000 --workers 1`
- GPU path: install CUDA/cuDNN images and vLLM; set `LLM_ENGINE=vllm`

### CI/CD (GitHub Actions)

- Lint & test → build image → push → deploy
- Cache wheels and models across builds to cut cold start

### HF Spaces

- Gradio demo consuming `/ws/audio` or REST
- CPU only; choose smallest models; pre‑warm on startup

### Fly.io

- `fly launch` with `fly.toml` setting `services.concurrency` and CPU/GPU size
- Use volumes for weight cache; health checks for `/healthz`

### Kubernetes (optional)

- Pods: gateway, asr, llm, tts (or single pod for latency)
- Node pools: GPU for LLM/TTS, CPU for ASR
- HPA on concurrency & latency metrics

### Secrets and Config

- Store API keys, tokens in Fly secrets or Kubernetes Secrets
- SBOM and image signing (cosign); scan images (Trivy)

### Zero-Budget Optimizations
- Use Colab/Kaggle for GPU training/testing (free tiers with time limits).
- HF Spaces: Limit to CPU, small models; use persistent storage for weights.
- Fly.io: Stay under free allowances; monitor with `fly status`; auto-scale down.
- Estimated costs: MVP demo ~$0/mo; light prod (10 users/day) ~$5-10/mo on Fly.io paid if exceeding free.


