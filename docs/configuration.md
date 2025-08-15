# Configuration Guide (Per .cursorrules)

## Environment Variables

**All settings use INZWA_ prefix per .cursorrules**. Copy `.env.example` to `.env` and customize.

## Configuration Options

### Core Settings (Minimal)

```bash
# Per .cursorrules template
INZWA_DEBUG=false
INZWA_CORS_ALLOWED_ORIGINS=http://localhost:7860  # No wildcards!
INZWA_REQUEST_TIMEOUT_S=5.0
INZWA_MAX_TEXT_CHARS=400
INZWA_MAX_AUDIO_SECONDS=20

# Auth (optional in dev)
INZWA_REQUIRE_AUTH=false
INZWA_API_KEY=your-secret-key-here
```

### ASR Settings

```bash
# Ultra-light per .cursorrules
INZWA_ASR_ENGINE=faster-whisper   # or whisper.cpp
INZWA_ASR_MODEL=small             # base/small only (no large models)
INZWA_ASR_DEVICE=cpu
INZWA_ASR_COMPUTE_TYPE=int8       # INT8 quantization mandatory

# VAD
INZWA_ASR_VAD_ENABLED=true
INZWA_ASR_VAD_MODEL=silero        # or webrtc
```

### LLM Settings

```bash
# CPU-first per .cursorrules
INZWA_LLM_ENGINE=llama-cpp        # vLLM only as flag
INZWA_LLM_MODEL=mistral-2b-shona-lora
INZWA_LLM_DEVICE=cpu

# Constrained generation
INZWA_LLM_MAX_TOKENS=512          # Reduce under load
INZWA_LLM_TEMPERATURE=0.7
INZWA_LLM_CONTEXT_SIZE=4096

# Quantization mandatory
INZWA_LLM_QUANTIZATION=Q4_K_M     # Q4/Q5 only
INZWA_LLM_GPU_LAYERS=0

# Model Path (optional)
LLM_MODEL_PATH=/path/to/model.gguf
```

### TTS (Text-to-Speech) Settings

```bash
# Engine Configuration
TTS_ENGINE=coqui-vits-lite
TTS_MODEL=shona_female_a
TTS_DEVICE=cpu           # Options: cpu, cuda
TTS_USE_ONNX=true       # Use ONNX for faster inference

# Audio Output
AUDIO_OUT_CODEC=opus     # Options: opus, pcm16

# Model Path (optional)
TTS_MODEL_PATH=/path/to/tts/model
```

### Performance Settings

```bash
# Per .cursorrules limits
INZWA_MAX_CONCURRENT_SESSIONS=50
INZWA_SESSION_TIMEOUT_SECONDS=60

# Streaming (mandatory)
INZWA_STREAMING_ENABLED=true
INZWA_AUDIO_CHUNK_MS=20          # 20-40ms chunks only
INZWA_BACKPRESSURE_THRESHOLD=8   # Max queue size per .cursorrules

# Hardware Optimization
GPU_POLICY=prefer       # Options: prefer, require, disable
QUANTIZATION_LEVEL=aggressive  # Options: aggressive, moderate, none
```

### Observability Settings

```bash
# Logging
LOG_LEVEL=INFO          # Options: DEBUG, INFO, WARNING, ERROR

# Metrics
METRICS_ENABLED=true
TRACING_ENABLED=false

# Data Collection (opt-in)
DATA_COLLECTION_ENABLED=false
DATA_RETENTION_DAYS=7
```

### Model Management

```bash
# Model Storage
MODEL_CACHE_DIR=./models
MODEL_WARMUP_ON_START=false

# Hugging Face Integration
HUGGINGFACE_HUB_TOKEN=your-token-here
```

### Feature Flags

```bash
# Advanced Features
ENABLE_WEBRTC=true
ENABLE_WEBSOCKET=true
ENABLE_RAG=false
ENABLE_SAFETY_FILTERS=true
ENABLE_SPECULATIVE_DECODING=false
ENABLE_EDGE_INFERENCE=false
```

## Configuration Profiles

### Development Profile

```bash
# Optimized for development
LOG_LEVEL=DEBUG
ASR_MODEL=tiny
LLM_ENGINE=llama-cpp
LLM_QUANTIZATION=Q4_K_M
MAX_CONCURRENT_SESSIONS=5
MODEL_WARMUP_ON_START=false
```

### Production Profile

```bash
# Optimized for production
LOG_LEVEL=INFO
ASR_MODEL=small
ASR_DEVICE=cuda
LLM_ENGINE=vllm
TTS_USE_ONNX=true
MAX_CONCURRENT_SESSIONS=100
MODEL_WARMUP_ON_START=true
ENABLE_SAFETY_FILTERS=true
```

### Edge/Mobile Profile

```bash
# Optimized for edge devices
ASR_MODEL=tiny
ASR_COMPUTE_TYPE=int8
LLM_QUANTIZATION=Q4_0
LLM_MAX_TOKENS=256
TTS_USE_ONNX=true
MAX_CONCURRENT_SESSIONS=1
ENABLE_EDGE_INFERENCE=true
```

## Advanced Configuration

### Custom Model Paths

To use custom-trained models:

1. Place models in the `MODEL_CACHE_DIR` directory
2. Set the appropriate `*_MODEL_PATH` environment variable
3. Ensure model format compatibility

### Multi-GPU Setup

For multi-GPU systems:

```bash
# Distribute models across GPUs
ASR_DEVICE=cuda:0
LLM_DEVICE=cuda:1
TTS_DEVICE=cuda:0
```

### Memory Optimization

For low-memory systems:

```bash
# Reduce memory usage
LLM_CONTEXT_SIZE=2048
LLM_GPU_LAYERS=10  # Partial GPU offload
MAX_CONCURRENT_SESSIONS=10
BACKPRESSURE_THRESHOLD=50
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce model sizes or use more aggressive quantization
2. **High Latency**: Enable GPU, use smaller models, or adjust chunk sizes
3. **Model Loading Fails**: Check paths and ensure models are downloaded
4. **WebSocket Disconnects**: Increase `SESSION_TIMEOUT_SECONDS`

### Environment Variable Precedence

1. Command-line arguments (highest priority)
2. Environment variables
3. `.env` file
4. Default values (lowest priority)
