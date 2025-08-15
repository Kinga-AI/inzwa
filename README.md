# Inzwa - Real-Time Shona Voice Assistant 🇿🇼

**Inzwa** ("Listen" in Shona) is a state-of-the-art, open-source, real-time speech-to-speech AI assistant that speaks fluent Shona. Built with cutting-edge ML models and optimized for ultra-low latency (<1s end-to-end), it enables natural voice conversations in Shona.

## 🌟 Features

- **Real-time Streaming**: Full duplex audio streaming with <500ms time-to-first-word
- **Native Shona**: Culturally-aware, fluent Shona understanding and generation
- **Ultra-Low Latency**: Optimized pipeline achieving <1s end-to-end response time
- **Zero-Budget Ready**: Runs on CPU, deployable on free tiers (HF Spaces, Fly.io)
- **Production-Grade**: WebSocket/WebRTC support, session management, observability
- **Modular Architecture**: Swappable ASR/LLM/TTS engines for A/B testing
- **Privacy-First**: Opt-in data collection, PII redaction, on-device capable

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
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

# List available models
curl http://localhost:8000/v1/models

# WebSocket streaming (use a WebSocket client)
ws://localhost:8000/ws/audio
```

## 🏗️ Architecture

Inzwa uses a streaming pipeline architecture:

```
Voice Input → ASR (Whisper) → LLM (Mistral/Gemma) → TTS (Coqui) → Voice Output
                ↑                                              ↓
                └──────────── Real-time Streaming ────────────┘
```

### Core Components

- **ASR**: Whisper (faster-whisper/whisper.cpp) with VAD for speech recognition
- **LLM**: Mistral-2B/Gemma-2B with LoRA fine-tuning for Shona conversations
- **TTS**: Coqui VITS-lite trained on Shona voice data
- **Orchestrator**: Async pipeline managing streaming, backpressure, and state

## 📚 Documentation

- [Architecture](docs/architecture.md) - System architecture and design principles
- [System Design](docs/system-design.md) - Detailed component interactions
- [Technical Specs](docs/technical-specs.md) - API specifications and schemas
- [API Reference](docs/api.md) - REST and WebSocket API documentation
- [Latency Engineering](docs/latency.md) - Performance optimization strategies
- [MLOps](docs/mlops.md) - Model training and deployment pipelines
- [Data Strategy](docs/data.md) - Data collection and preparation
- [Security & Privacy](docs/security-privacy.md) - Security measures and privacy policies
- [Deployment Guide](docs/deployment.md) - Deployment instructions for various platforms
- [Roadmap](docs/roadmap.md) - Development roadmap and milestones

## 🎯 Use Cases

- **Education**: Shona language learning and practice
- **Accessibility**: Voice interface for Shona speakers
- **Customer Service**: Automated Shona customer support
- **Cultural Preservation**: Documenting and promoting Shona language
- **Healthcare**: Medical assistance in native language

## 🤝 Contributing

We welcome contributions! Areas where help is needed:

- **Data Collection**: Recording Shona speech samples
- **Model Training**: Fine-tuning models for better Shona support
- **Testing**: Adding test coverage and edge cases
- **Documentation**: Improving docs and adding examples
- **Frontend**: Building web/mobile clients

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 🔧 Configuration

See [Configuration Guide](docs/configuration.md) for detailed configuration options.

## 🧪 Development

See [Development Guide](docs/development.md) for development setup and workflows.

## 📊 Performance

| Metric | Target | Current |
|--------|--------|--------|
| Time to First Word | <500ms | ~600ms |
| End-to-End Latency | <1s | ~1.2s |
| Word Error Rate | <12% | ~15% |

## 🗺️ Roadmap

- [x] Core pipeline implementation
- [x] WebSocket/REST API
- [x] Docker containerization
- [ ] Shona model fine-tuning
- [ ] WebRTC support
- [ ] Edge deployment (WASM)
- [ ] Mobile SDK
- [ ] Multi-speaker support
- [ ] Ndebele language support

## 📄 License

Apache-2.0 License. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- OpenAI Whisper for ASR foundation
- Mistral AI for LLM base models  
- Coqui AI for TTS framework
- The Shona-speaking community for language support

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/kinga-ai/inzwa/issues)
- **Security**: security@kinga.ai
- **Community**: [Discord](https://discord.gg/kinga-ai) (coming soon)

---

**Inzwa** - Bringing voice AI to Shona speakers worldwide 🌍
