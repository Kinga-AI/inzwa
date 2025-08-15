# Development Guide

## Prerequisites

- Python 3.10 or higher
- Poetry for dependency management
- Git for version control
- (Optional) Docker for containerized development
- (Optional) CUDA-capable GPU for accelerated inference

## Setting Up Development Environment

### 1. Clone the Repository

```bash
git clone https://github.com/kinga-ai/inzwa.git
cd inzwa
```

### 2. Install Poetry

```bash
# Using pip
pip install poetry

# Or using the official installer
curl -sSL https://install.python-poetry.org | python3 -
```

### 3. Install Dependencies

```bash
# Install all dependencies including dev tools
poetry install --extras full

# Or install minimal dependencies
poetry install

# Or install specific components
poetry install --extras "asr llm tts"
```

### 4. Activate Virtual Environment

```bash
# Activate the Poetry-managed virtual environment
poetry shell

# Or run commands with poetry run
poetry run python --version
```

## Running Tests

### Unit Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src/inzwa --cov-report=html

# Run specific test file
poetry run pytest tests/test_api.py

# Run with verbose output
poetry run pytest -v

# Run tests in parallel
poetry run pytest -n auto
```

### Integration Tests

```bash
# Run integration tests
poetry run pytest tests/integration/ -v

# Test WebSocket connections
poetry run pytest tests/test_websocket.py
```

### Performance Tests

```bash
# Run latency benchmarks
poetry run python tests/benchmarks/test_latency.py

# Load testing
poetry run locust -f tests/load/locustfile.py
```

## Code Quality

### Linting

```bash
# Run ruff linter
poetry run ruff check src/

# Auto-fix issues
poetry run ruff check src/ --fix

# Check specific file
poetry run ruff check src/inzwa/api/app.py
```

### Formatting

```bash
# Format code with black
poetry run black src/

# Check formatting without changes
poetry run black --check src/

# Format specific file
poetry run black src/inzwa/api/app.py
```

### Type Checking

```bash
# Run mypy type checker
poetry run mypy src/inzwa

# Ignore missing imports
poetry run mypy src/inzwa --ignore-missing-imports

# Strict mode
poetry run mypy src/inzwa --strict
```

## Development Workflow

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

Follow the project structure:
- API endpoints go in `src/inzwa/api/routers/`
- ML engines go in respective modules (`asr/`, `llm/`, `tts/`)
- Shared utilities in `src/inzwa/utils/`
- Tests mirror the source structure

### 3. Write Tests

```python
# tests/test_your_feature.py
import pytest
from inzwa.your_module import your_function

def test_your_function():
    result = your_function("input")
    assert result == "expected_output"

@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result is not None
```

### 4. Run Pre-commit Checks

```bash
# Run all checks
make lint
make test

# Or manually
poetry run black src/
poetry run ruff check src/
poetry run mypy src/inzwa
poetry run pytest
```

### 5. Commit Changes

```bash
git add .
git commit -m "feat: add your feature description"
```

Follow conventional commits:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation
- `test:` for tests
- `refactor:` for refactoring
- `perf:` for performance improvements

### 6. Push and Create PR

```bash
git push origin feature/your-feature-name
```

## Debugging

### Running in Debug Mode

```bash
# Set debug environment variable
export LOG_LEVEL=DEBUG

# Run with debugger
poetry run python -m debugpy --listen 5678 --wait-for-client -m uvicorn inzwa.api.app:app --reload
```

### Using IPython for Interactive Debugging

```bash
# Start IPython shell with project context
poetry run ipython

# In IPython
from inzwa.asr import ASREngine
engine = ASREngine()
# Test your code interactively
```

### Logging

```python
from inzwa.telemetry import get_logger

logger = get_logger(__name__)

logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
```

## Docker Development

### Building Docker Image

```bash
# Build development image
docker build -t inzwa:dev .

# Build with specific Python version
docker build --build-arg PYTHON_VERSION=3.11 -t inzwa:dev .
```

### Running in Docker

```bash
# Run with volume mounting for development
docker run -it --rm \
  -v $(pwd):/app \
  -p 8000:8000 \
  inzwa:dev

# Run with GPU support
docker run --gpus all -it --rm \
  -p 8000:8000 \
  inzwa:dev
```

### Docker Compose Development

```yaml
# docker-compose.dev.yml
version: '3.8'
services:
  inzwa:
    build: .
    volumes:
      - .:/app
      - ./models:/app/models
    ports:
      - "8000:8000"
    environment:
      - LOG_LEVEL=DEBUG
      - MODEL_WARMUP_ON_START=false
    command: uvicorn inzwa.api.app:app --reload --host 0.0.0.0
```

## Model Development

### Training Custom Models

```bash
# Fine-tune LLM
poetry run python scripts/train_llm.py \
  --base-model mistral-2b \
  --dataset data/shona_conversations.json \
  --output models/llm/shona-lora

# Train TTS
poetry run python scripts/train_tts.py \
  --dataset data/shona_audio \
  --output models/tts/shona_voice
```

### Testing Models

```python
# Test ASR model
from inzwa.asr import ASREngine

engine = ASREngine()
result = await engine.transcribe_batch(audio_bytes)
print(f"Transcription: {result['text']}")

# Test LLM
from inzwa.llm import LLMEngine

engine = LLMEngine()
response = await engine.generate_batch([
    {"role": "user", "content": "Mhoro"}
])
print(f"Response: {response}")
```

## Performance Profiling

### CPU Profiling

```bash
# Profile with cProfile
poetry run python -m cProfile -o profile.stats scripts/benchmark.py

# Analyze results
poetry run python -m pstats profile.stats
```

### Memory Profiling

```bash
# Install memory profiler
poetry add --dev memory-profiler

# Run with memory profiling
poetry run mprof run python scripts/benchmark.py
poetry run mprof plot
```

### GPU Profiling

```bash
# Using NVIDIA tools
nvidia-smi dmon -s um -d 1

# PyTorch profiler
poetry run python scripts/profile_gpu.py
```

## Continuous Integration

The project uses GitHub Actions for CI/CD. See `.github/workflows/ci.yml`.

### Local CI Testing

```bash
# Install act for local GitHub Actions testing
brew install act  # macOS
# or
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash  # Linux

# Run CI locally
act -j test
```

## Troubleshooting

### Common Development Issues

1. **Poetry lock file out of sync**
   ```bash
   poetry lock --no-update
   poetry install
   ```

2. **Import errors**
   ```bash
   # Ensure you're in the Poetry environment
   poetry shell
   # Or use poetry run
   poetry run python your_script.py
   ```

3. **Model download issues**
   ```bash
   # Clear cache and re-download
   rm -rf models/
   poetry run python scripts/download_models.py
   ```

4. **Port already in use**
   ```bash
   # Find and kill process using port 8000
   lsof -i :8000
   kill -9 <PID>
   ```

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines.
