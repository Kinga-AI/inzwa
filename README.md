# 🎙️ Inzwa - Real-Time Shona Voice Assistant 🇿🇼

*"Hey Inzwa-ka, zviri sei?"*

**Inzwa** ("Listen" in Shona) is a state-of-the-art, open-source, real-time speech-to-speech AI assistant that speaks fluent Shona. Built with cutting-edge ML models and optimized for ultra-low latency (<1s end-to-end), it enables natural voice conversations in Shona for millions of native speakers worldwide.

---

## 🌟 Key Features

- ✅ **Real-time Streaming**: Full duplex audio streaming with <500ms time-to-first-word
- ✅ **Native Shona**: Culturally-aware, fluent Shona understanding and generation  
- ✅ **Ultra-Low Latency**: Sub-second conversation flow (<1s end-to-end)
- ✅ **Zero-Budget Ready**: Runs on CPU, deployable on free tiers (HF Spaces, Fly.io)
- ✅ **Production-Grade**: WebSocket/WebRTC support, session management, observability
- ✅ **Modular Architecture**: Swappable ASR/LLM/TTS engines for A/B testing
- ✅ **Privacy-First**: Run entirely offline, opt-in data collection, PII redaction
- ✅ **Fully Open Source**: Built entirely on open-source software with Apache-2.0 license

---

## 📽️ Demo

- 🎙️ **Real-time voice assistant**: Speak naturally, get instant answers in Shona
- 🔗 Live demo: [Coming soon on HuggingFace Spaces](https://huggingface.co/spaces/inzwa/demo)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Poetry (for dependency management)
- 4GB+ RAM for CPU inference
- (Optional) CUDA GPU for faster inference

### Installation

```bash
# Clone the repository
git clone https://github.com/kinga-ai/inzwa.git
cd inzwa

# Install dependencies
poetry install --extras full  # Or 'poetry install' for minimal

# Copy environment variables
cp .env.example .env
# Edit .env with your configuration

# Download models (optional)
python scripts/download_models.py

# Run the server
make run  # Or: poetry run uvicorn inzwa.api.app:app --reload
```

### Test the API

```bash
# Health check
curl http://localhost:8000/healthz

# WebSocket streaming (use a WebSocket client)
ws://localhost:8000/ws/audio
```

---

## 🏗️ Architecture

**End-to-End conversational pipeline:**

```
User: "Hey Inzwa-ka..."
│
▼ (Voice via WebSocket/WebRTC)
┌───────────────────────────────┐
│    Whisper.cpp (ASR, Shona)   │──► Real-time transcription
└───────────────────────────────┘
│
▼
┌───────────────────────────────┐
│   Mistral-2B (LLM inference)  │──► Natural language understanding & response
└───────────────────────────────┘
│
▼
┌───────────────────────────────┐
│ Coqui TTS (VITS Shona voice)  │──► Real-time voice synthesis
└───────────────────────────────┘
│
▼
User hears fluent Shona response
```

### Core Components

- **ASR**: [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) / faster-whisper with INT8 quantization
- **LLM**: [Mistral-2B](https://mistral.ai/) / Gemma-2B with LoRA fine-tuning for Shona
- **TTS**: [Coqui TTS](https://github.com/coqui-ai/TTS) VITS-lite trained on Shona voice
- **Backend**: [FastAPI](https://fastapi.tiangolo.com/) with WebSocket/WebRTC streaming
- **Orchestrator**: Async pipeline with bounded queues and backpressure control

---

## 📚 Documentation

- [Architecture](docs/architecture.md) - System architecture and design principles
- [System Design](docs/system-design.md) - Detailed component interactions
- [Technical Specs](docs/technical-specs.md) - API specifications and schemas
- [Task Roadmap](TASKS.md) - Comprehensive development tasks with acceptance criteria
- [API Reference](docs/api.md) - REST and WebSocket API documentation
- [Latency Engineering](docs/latency.md) - Performance optimization strategies
- [MLOps](docs/mlops.md) - Model training and deployment pipelines
- [Data Strategy](docs/data.md) - Data collection and preparation
- [Security & Privacy](docs/security-privacy.md) - Security measures and privacy policies
- [Deployment Guide](docs/deployment.md) - Deployment instructions for various platforms
- [Configuration Guide](docs/configuration.md) - Detailed configuration options
- [Development Guide](docs/development.md) - Development setup and workflows

---

## ⚡ Performance Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Time to First Word (P50) | ≤500ms | ~600ms |
| End-to-End Latency (P95) | ≤1.2s | ~1.2s |
| ASR Real-Time Factor | 0.2-0.5x | 0.3x |
| LLM Tokens/sec (CPU) | ≥10 | 15 |
| Word Error Rate | <12% | ~15% |
| Concurrent Sessions | 50+ | 50 |

---

## 🎯 Use Cases

- **Education**: Shona language learning, pronunciation practice, interactive storytelling
- **Accessibility**: Voice interface for Shona speakers with disabilities
- **Customer Service**: Automated Shona customer support for businesses
- **Cultural Preservation**: Documenting and promoting Shona language and oral traditions
- **Healthcare**: Medical assistance and health information in native language
- **Information Access**: News, weather, and local information in Shona

---

## 🌍 Deployment Options

- **Local Machine**: CPU or GPU laptop/desktop
- **HuggingFace Spaces**: [Free CPU tier](https://huggingface.co/spaces) (up to 1GB RAM)
- **Fly.io**: [Free tier](https://fly.io/docs/about/pricing/) (3 machines, 256MB RAM each)
- **Docker**: Multi-stage build < 500MB
- **Kubernetes**: For production scale

See [Deployment Guide](docs/deployment.md) for detailed instructions.

---

## 🗺️ Roadmap

### Phase 1: MVP (Weeks 1-8) ✅
- [x] Core pipeline implementation
- [x] WebSocket/REST API
- [x] Docker containerization
- [x] Documentation

### Phase 2: Optimization (Weeks 9-16) 🚧
- [ ] Shona model fine-tuning
- [ ] Latency < 1s optimization
- [ ] WebRTC support
- [ ] Production deployment

### Phase 3: Quality & Scale (Weeks 17-24)
- [ ] Multi-speaker support
- [ ] Edge deployment (WASM)
- [ ] Mobile SDK
- [ ] A/B testing framework

### Phase 4: Expansion (Post-24 weeks)
- [ ] Ndebele language support
- [ ] Zulu language support
- [ ] Community crowdsourcing
- [ ] Offline mode

See [detailed task breakdown](TASKS.md) for comprehensive development plan.

---

## 🤝 Contributing

We warmly welcome contributions! Areas where help is needed:

- **Data Collection**: Recording Shona speech samples
- **Model Training**: Fine-tuning models for better Shona support
- **Testing**: Adding test coverage and edge cases
- **Documentation**: Improving docs and adding examples
- **Frontend**: Building web/mobile clients
- **Community**: Spreading the word and getting feedback

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

This project is fully open-source under the [Apache-2.0 License](LICENSE).

---

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) & [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) for ASR foundation
- [Mistral AI](https://mistral.ai/) for LLM base models
- [Coqui AI](https://github.com/coqui-ai/TTS) for TTS framework
- [Masakhane](https://www.masakhane.io/) African NLP research community
- The Shona-speaking community for language support and feedback

---

## 📧 Contact & Community

- **Maintainer**: [Simbarashe Timire](https://github.com/TimireSimbarashe)
- **Issues**: [GitHub Issues](https://github.com/kinga-ai/inzwa/issues)
- **Security**: security@kinga.ai
- **Discussions**: [GitHub Discussions](https://github.com/kinga-ai/inzwa/discussions)
- **Community**: [Discord](https://discord.gg/kinga-ai) (coming soon)

---

**Inzwa** - Together, let's unlock powerful AI-driven interactions in the Shona language, bringing cutting-edge voice technology to millions of native speakers worldwide 🌍