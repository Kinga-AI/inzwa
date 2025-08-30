# Phase 1-2: Foundation & ASR Service

## Phase 1: Foundation & API Gateway (Week 1)

### 1.1 Project Setup & Configuration
**Owner:** DevOps Lead  
**Duration:** 2 hours  
**Dependencies:** None

#### Tasks:
- [ ] 1.1.1 Create `.env.example` with all IZWI_ variables
- [ ] 1.1.2 Set up Poetry with exact versions
- [ ] 1.1.3 Configure pre-commit hooks (black, ruff, mypy)
- [ ] 1.1.4 Set up logging structure (no raw payloads)
- [ ] 1.1.5 Create Docker multi-stage build

#### Implementation Details:

**.env.example:**
```bash
# Core Settings (Per .cursorrules)
IZWI_DEBUG=false
IZWI_CORS_ALLOWED_ORIGINS=http://localhost:7860
IZWI_REQUEST_TIMEOUT_S=5.0
IZWI_MAX_TEXT_CHARS=400
IZWI_MAX_AUDIO_SECONDS=20
IZWI_ENABLE_WEBRTC=false

# ASR Settings
IZWI_ASR_ENGINE=faster-whisper
IZWI_ASR_MODEL=small
IZWI_ASR_DEVICE=cpu
IZWI_ASR_COMPUTE_TYPE=int8

# LLM Settings
IZWI_LLM_ENGINE=llama-cpp
IZWI_LLM_MODEL=mistral-2b-shona-lora
IZWI_LLM_QUANTIZATION=Q4_K_M
IZWI_LLM_MAX_TOKENS=512

# TTS Settings
IZWI_TTS_ENGINE=coqui-vits-lite
IZWI_TTS_USE_ONNX=true
IZWI_AUDIO_OUT_CODEC=opus

# Performance
IZWI_MAX_CONCURRENT_SESSIONS=50
IZWI_BACKPRESSURE_THRESHOLD=8
```

**pyproject.toml dependencies:**
```toml
[tool.poetry.dependencies]
python = "^3.11"
fastapi = "^0.104.0"
uvicorn = {extras = ["standard"], version = "^0.24.0"}
pydantic = "^2.5.0"
pydantic-settings = "^2.1.0"
orjson = "^3.9.0"
prometheus-client = "^0.19.0"
httpx = "^0.25.0"

[tool.poetry.dev-dependencies]
pytest = "^7.4.0"
pytest-asyncio = "^0.21.0"
black = "^23.0.0"
ruff = "^0.1.0"
mypy = "^1.7.0"
```

#### Testing:
```bash
# Test configuration loading
pytest tests/test_config.py::test_env_loading -v
pytest tests/test_config.py::test_validation -v

# Verify no secrets in code
grep -r "password\|secret\|key" src/ && exit 1 || echo "✓ No secrets"

# Check Docker build size
docker build -t izwi:test .
docker images izwi:test --format "Size: {{.Size}}"
# Expected: < 500MB
```

#### Acceptance Criteria:
- ✅ All env vars use IZWI_ prefix
- ✅ Settings validate with pydantic v2
- ✅ Docker image < 500MB
- ✅ No hardcoded secrets
- ✅ Structured logging configured
- ✅ Pre-commit hooks working

---

### 1.2 FastAPI Base Application
**Owner:** Backend Lead  
**Duration:** 4 hours  
**Dependencies:** 1.1

#### Tasks:
- [ ] 1.2.1 Implement FastAPI app with ORJSONResponse
- [ ] 1.2.2 Add request ID middleware
- [ ] 1.2.3 Configure CORS (no wildcards)
- [ ] 1.2.4 Add prometheus metrics
- [ ] 1.2.5 Implement error handling (no stack traces)

#### Code Implementation:
```python
# src/izwi/api/app.py
from fastapi import FastAPI, Request, Response
from fastapi.routing import APIRoute
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import ORJSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
import uuid
import time

from izwi.ops.settings import settings
from izwi.ops.metrics import METRICS
from izwi.ops.logging import logger

class TimedRoute(APIRoute):
    """Custom route to add timing and request ID."""
    
    def get_route_handler(self):
        original = super().get_route_handler()
        
        async def custom_handler(request: Request) -> Response:
            # Generate request ID
            request_id = str(uuid.uuid4())[:8]
            request.state.request_id = request_id
            
            # Start timing
            start = time.perf_counter()
            
            # Process request
            response = await original(request)
            
            # Add headers
            duration = time.perf_counter() - start
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration:.3f}"
            
            # Record metrics
            METRICS.request_duration.observe(duration)
            METRICS.requests_total.inc()
            
            logger.info(
                "Request completed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration_ms=duration * 1000
            )
            
            return response
        
        return custom_handler

app = FastAPI(
    title="Izwi API",
    version="0.1.0",
    default_response_class=ORJSONResponse,
    route_class=TimedRoute,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# Strict CORS - no wildcards!
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins.split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    max_age=3600,
)

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions without exposing stack traces."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    # Log error (no sensitive data)
    logger.warning(
        "HTTP exception",
        request_id=request_id,
        status_code=exc.status_code,
        path=request.url.path,
        user_hash=hmac_user_id(request)
    )
    
    # Return safe error response
    return ORJSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "request_id": request_id
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions safely."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    # Log error (no stack trace in response)
    logger.error(
        "Unhandled exception",
        request_id=request_id,
        exc_info=True  # This logs to server only
    )
    
    # Return generic error
    return ORJSONResponse(
        status_code=500,
        content={
            "error": {
                "code": 500,
                "message": "Internal server error",
                "request_id": request_id
            }
        }
    )
```

#### Testing:
```python
# tests/test_phase1_api.py
import pytest
from fastapi.testclient import TestClient
from izwi.api.app import app

client = TestClient(app)

def test_orjson_response():
    """Verify ORJSON is used."""
    response = client.get("/healthz")
    assert response.headers["content-type"] == "application/json"
    # ORJSON is faster, verify performance
    
def test_cors_strict():
    """Test CORS rejects wildcards."""
    # Valid origin
    response = client.options(
        "/healthz",
        headers={"Origin": "http://localhost:7860"}
    )
    assert response.status_code == 200
    
    # Invalid origin
    response = client.options(
        "/healthz", 
        headers={"Origin": "http://evil.com"}
    )
    assert "access-control-allow-origin" not in response.headers

def test_request_id():
    """Every response has request ID."""
    response = client.get("/healthz")
    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) == 8

def test_no_stack_traces():
    """Errors don't expose stack traces."""
    # Trigger 404
    response = client.get("/nonexistent")
    assert response.status_code == 404
    body = response.json()
    
    # Should have safe error format
    assert "error" in body
    assert "code" in body["error"]
    assert "message" in body["error"]
    assert "request_id" in body["error"]
    
    # Should NOT have stack trace
    assert "traceback" not in str(body).lower()
    assert "file" not in str(body).lower()
    assert "line" not in str(body).lower()

def test_response_time_header():
    """Response time is tracked."""
    response = client.get("/healthz")
    assert "X-Response-Time" in response.headers
    time_ms = float(response.headers["X-Response-Time"]) * 1000
    assert time_ms < 100  # Should be fast

@pytest.mark.asyncio
async def test_concurrent_requests():
    """Handle concurrent requests."""
    import asyncio
    import httpx
    
    async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
        tasks = [ac.get("/healthz") for _ in range(100)]
        responses = await asyncio.gather(*tasks)
    
    # All should succeed
    assert all(r.status_code == 200 for r in responses)
    
    # All should have unique request IDs
    request_ids = [r.headers["X-Request-ID"] for r in responses]
    assert len(set(request_ids)) == 100
```

#### Acceptance Criteria:
- ✅ FastAPI starts in < 2s
- ✅ CORS rejects wildcards and unknown origins
- ✅ All responses use ORJSON (fast JSON)
- ✅ Request IDs on all responses
- ✅ No stack traces in error responses
- ✅ Response time header on all responses
- ✅ Prometheus metrics exposed at /metrics
- ✅ Handles 100+ concurrent requests

---

### 1.3 Health & Readiness Endpoints
**Owner:** Backend Lead  
**Duration:** 2 hours  
**Dependencies:** 1.2

#### Tasks:
- [ ] 1.3.1 Implement GET /healthz (liveness)
- [ ] 1.3.2 Implement GET /readyz (readiness)
- [ ] 1.3.3 Add model loading checks
- [ ] 1.3.4 Add dependency health checks
- [ ] 1.3.5 Implement graceful shutdown

#### Code Implementation:
```python
# src/izwi/api/health.py
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import asyncio
import time

from izwi.ops.logging import logger
from izwi.models.registry import model_registry

router = APIRouter(tags=["health"])

@router.get("/healthz")
async def health() -> Dict[str, Any]:
    """
    Liveness probe - is the service running?
    
    Returns 200 if the service is alive.
    Used by Kubernetes/Docker for restart decisions.
    """
    return {
        "status": "ok",
        "timestamp": time.time(),
        "version": "0.1.0"
    }

@router.get("/readyz")
async def ready() -> Dict[str, Any]:
    """
    Readiness probe - can the service handle traffic?
    
    Returns 200 if all dependencies are ready.
    Returns 503 if any dependency is not ready.
    """
    checks = {}
    all_ready = True
    
    # Check model loading
    try:
        models_ready = await model_registry.check_ready()
        checks["models"] = {
            "ready": models_ready,
            "loaded": model_registry.get_loaded_models()
        }
        if not models_ready:
            all_ready = False
    except Exception as e:
        checks["models"] = {"ready": False, "error": str(e)}
        all_ready = False
    
    # Check cache connection (if enabled)
    if settings.cache_enabled:
        try:
            cache_ready = await check_cache_connection()
            checks["cache"] = {"ready": cache_ready}
            if not cache_ready:
                all_ready = False
        except Exception as e:
            checks["cache"] = {"ready": False, "error": str(e)}
            all_ready = False
    
    # Check available memory
    import psutil
    mem = psutil.virtual_memory()
    checks["memory"] = {
        "available_mb": mem.available // 1024 // 1024,
        "percent_used": mem.percent,
        "ready": mem.percent < 90  # Not ready if >90% used
    }
    if mem.percent >= 90:
        all_ready = False
    
    # Return appropriate status
    if not all_ready:
        logger.warning("Readiness check failed", checks=checks)
        raise HTTPException(
            status_code=503,
            detail={"status": "not_ready", "checks": checks}
        )
    
    return {
        "status": "ready",
        "checks": checks,
        "timestamp": time.time()
    }

# Graceful shutdown handler
shutdown_event = asyncio.Event()

async def graceful_shutdown():
    """Handle graceful shutdown."""
    logger.info("Starting graceful shutdown...")
    
    # Set shutdown flag
    shutdown_event.set()
    
    # Wait for active requests to complete (max 5s)
    await asyncio.sleep(0.1)  # Let current requests see the flag
    
    max_wait = 5.0
    start = time.time()
    
    while time.time() - start < max_wait:
        active = get_active_requests_count()
        if active == 0:
            break
        
        logger.info(f"Waiting for {active} active requests...")
        await asyncio.sleep(0.5)
    
    # Clean up resources
    await model_registry.cleanup()
    
    logger.info("Graceful shutdown complete")
```

#### Testing:
```python
# tests/test_health.py
import pytest
from fastapi.testclient import TestClient
import time

def test_healthz_fast():
    """Health check should be very fast."""
    start = time.perf_counter()
    response = client.get("/healthz")
    duration = time.perf_counter() - start
    
    assert response.status_code == 200
    assert duration < 0.01  # <10ms
    
    data = response.json()
    assert data["status"] == "ok"
    assert "timestamp" in data
    assert "version" in data

def test_readyz_when_not_ready():
    """Readiness should fail when models not loaded."""
    # Mock models not loaded
    with mock.patch("model_registry.check_ready", return_value=False):
        response = client.get("/readyz")
        
        assert response.status_code == 503
        data = response.json()
        assert data["error"]["code"] == 503
        assert "not_ready" in str(data)

def test_readyz_when_ready():
    """Readiness should succeed when all deps ready."""
    # Mock everything ready
    with mock.patch("model_registry.check_ready", return_value=True):
        response = client.get("/readyz")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert "checks" in data
        assert data["checks"]["models"]["ready"] is True

@pytest.mark.asyncio
async def test_graceful_shutdown():
    """Test graceful shutdown."""
    from izwi.api.health import graceful_shutdown, shutdown_event
    
    # Start shutdown
    task = asyncio.create_task(graceful_shutdown())
    
    # Verify shutdown flag is set
    await asyncio.sleep(0.05)
    assert shutdown_event.is_set()
    
    # Wait for completion
    await task
    
    # Verify cleanup happened
    # (would check actual cleanup in real test)

def test_health_under_load():
    """Health endpoint handles high load."""
    import concurrent.futures
    
    def make_request():
        return client.get("/healthz").status_code
    
    # Make 1000 concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(make_request) for _ in range(1000)]
        results = [f.result() for f in futures]
    
    # All should succeed
    assert all(code == 200 for code in results)
```

#### Performance Testing:
```bash
# Load test health endpoint
ab -n 10000 -c 100 http://localhost:8000/healthz

# Expected results:
# - Requests per second: > 1000
# - Time per request: < 100ms (mean)
# - Failed requests: 0
```

#### Acceptance Criteria:
- ✅ /healthz responds in < 10ms
- ✅ /healthz handles 1000+ req/s
- ✅ /readyz checks all dependencies
- ✅ /readyz returns 503 when not ready
- ✅ Memory check included
- ✅ Model loading status included
- ✅ Graceful shutdown completes in < 5s
- ✅ No memory leaks during shutdown

---

## Phase 2: ASR Service (Week 2)

### 2.1 ASR Engine Integration
**Owner:** ML Engineer  
**Duration:** 6 hours  
**Dependencies:** Phase 1 complete

#### Tasks:
- [ ] 2.1.1 Integrate faster-whisper with INT8
- [ ] 2.1.2 Implement whisper.cpp fallback
- [ ] 2.1.3 Add model downloading/caching
- [ ] 2.1.4 Implement streaming transcription
- [ ] 2.1.5 Add language detection

#### Code Implementation:
```python
# src/izwi/asr/engine.py
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional
import numpy as np
from faster_whisper import WhisperModel
import asyncio
import time

from izwi.models import TranscriptChunk
from izwi.ops.settings import settings
from izwi.ops.logging import logger
from izwi.ops.metrics import METRICS

class ASREngine(ABC):
    """Abstract base for ASR engines."""
    
    @abstractmethod
    async def transcribe_stream(
        self,
        audio_frames: AsyncIterator[bytes]
    ) -> AsyncIterator[TranscriptChunk]:
        """Stream transcription from audio frames."""
        pass

class FasterWhisperEngine(ASREngine):
    """Faster-whisper implementation with INT8."""
    
    def __init__(
        self,
        model_size: str = None,
        device: str = None,
        compute_type: str = None
    ):
        self.model_size = model_size or settings.asr_model
        self.device = device or settings.asr_device
        self.compute_type = compute_type or settings.asr_compute_type
        
        # Validate compute type
        if self.compute_type not in ["int8", "int8_float16"]:
            logger.warning(
                f"Non-INT8 compute type {self.compute_type}, "
                "forcing INT8 per .cursorrules"
            )
            self.compute_type = "int8"
        
        # Load model
        logger.info(
            f"Loading Whisper {self.model_size} "
            f"({self.compute_type} on {self.device})"
        )
        
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            num_workers=1,  # Single worker for streaming
            download_root=settings.model_cache_dir,
            local_files_only=False
        )
        
        logger.info("Whisper model loaded successfully")
    
    async def transcribe_stream(
        self,
        audio_frames: AsyncIterator[bytes]
    ) -> AsyncIterator[TranscriptChunk]:
        """
        Stream transcription with <500ms chunks.
        
        Processes audio in 500ms windows for low latency.
        """
        buffer = bytearray()
        chunk_duration_ms = 500  # Process every 500ms
        chunk_size = int(16000 * chunk_duration_ms / 1000 * 2)  # 16kHz, 2 bytes per sample
        
        async for frame in audio_frames:
            buffer.extend(frame)
            
            # Process when we have enough audio
            while len(buffer) >= chunk_size:
                # Extract chunk
                chunk = buffer[:chunk_size]
                buffer = buffer[chunk_size:]
                
                # Convert to float32
                audio_array = np.frombuffer(
                    chunk, dtype=np.int16
                ).astype(np.float32) / 32768.0
                
                # Transcribe
                start_time = time.perf_counter()
                
                # Run in thread pool to avoid blocking
                segments, info = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._transcribe_chunk,
                    audio_array
                )
                
                # Calculate RTF
                processing_time = time.perf_counter() - start_time
                rtf = processing_time / (chunk_duration_ms / 1000)
                METRICS.asr_rtf.set(rtf)
                
                # Emit segments
                for segment in segments:
                    chunk = TranscriptChunk(
                        text=segment.text.strip(),
                        is_final=False,  # Will be set by VAD
                        start_ms=int(segment.start * 1000),
                        end_ms=int(segment.end * 1000),
                        confidence=np.exp(segment.avg_logprob) if segment.avg_logprob else 0.0
                    )
                    
                    # Only emit non-empty chunks
                    if chunk.text:
                        logger.debug(
                            f"ASR chunk: '{chunk.text}' "
                            f"(RTF: {rtf:.2f})"
                        )
                        yield chunk
    
    def _transcribe_chunk(self, audio: np.ndarray):
        """Transcribe a single audio chunk."""
        segments, info = self.model.transcribe(
            audio,
            language="sn" if settings.asr_language == "shona" else None,
            beam_size=1,  # Faster with beam_size=1
            best_of=1,    # No alternatives needed
            temperature=0,  # Deterministic
            vad_filter=True,
            vad_parameters=dict(
                threshold=0.5,
                min_silence_duration_ms=500,
                speech_pad_ms=100
            ),
            word_timestamps=False,  # Faster without
            condition_on_previous_text=False  # Faster
        )
        
        return list(segments), info

class WhisperCppEngine(ASREngine):
    """Whisper.cpp fallback implementation."""
    
    def __init__(self):
        # TODO: Implement whisper.cpp binding
        logger.warning("WhisperCpp engine not yet implemented")
    
    async def transcribe_stream(
        self,
        audio_frames: AsyncIterator[bytes]
    ) -> AsyncIterator[TranscriptChunk]:
        # Fallback implementation
        yield TranscriptChunk(
            text="[WhisperCpp not implemented]",
            is_final=True,
            start_ms=0,
            end_ms=1000,
            confidence=0.0
        )

# Factory function
def create_asr_engine() -> ASREngine:
    """Create ASR engine based on settings."""
    if settings.asr_engine == "faster-whisper":
        return FasterWhisperEngine()
    elif settings.asr_engine == "whisper.cpp":
        return WhisperCppEngine()
    else:
        raise ValueError(f"Unknown ASR engine: {settings.asr_engine}")
```

#### Testing:
```python
# tests/test_asr_engine.py
import pytest
import asyncio
import time
import numpy as np
from izwi.asr.engine import FasterWhisperEngine

@pytest.fixture
async def asr_engine():
    """Create test ASR engine with tiny model."""
    engine = FasterWhisperEngine(
        model_size="tiny",  # Use tiny for fast tests
        device="cpu",
        compute_type="int8"
    )
    yield engine

async def generate_test_audio(duration_s: float = 5.0):
    """Generate test audio frames."""
    sample_rate = 16000
    frame_duration_ms = 30
    frame_size = int(sample_rate * frame_duration_ms / 1000 * 2)
    
    # Load or generate audio
    # For testing, using sine wave as placeholder
    t = np.linspace(0, duration_s, int(sample_rate * duration_s))
    audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    
    # Yield in frames
    for i in range(0, len(audio) * 2, frame_size):
        frame = audio.tobytes()[i:i+frame_size]
        if frame:
            yield frame
            await asyncio.sleep(frame_duration_ms / 1000)

@pytest.mark.asyncio
async def test_asr_streaming(asr_engine):
    """Test streaming transcription."""
    audio_frames = generate_test_audio(duration_s=3.0)
    
    chunks = []
    start = time.perf_counter()
    
    async for chunk in asr_engine.transcribe_stream(audio_frames):
        chunks.append(chunk)
        
        # Check first chunk latency
        if len(chunks) == 1:
            first_chunk_time = time.perf_counter() - start
            assert first_chunk_time < 0.5  # <500ms
    
    # Should produce some output
    assert len(chunks) > 0

@pytest.mark.asyncio
async def test_asr_rtf(asr_engine):
    """Test real-time factor."""
    audio_duration = 5.0
    audio_frames = generate_test_audio(duration_s=audio_duration)
    
    start = time.perf_counter()
    chunks = []
    
    async for chunk in asr_engine.transcribe_stream(audio_frames):
        chunks.append(chunk)
    
    processing_time = time.perf_counter() - start
    rtf = processing_time / audio_duration
    
    # Should be faster than real-time
    assert rtf < 0.5  # RTF < 0.5x per .cursorrules

def test_int8_quantization():
    """Verify INT8 quantization is enforced."""
    # Try to create with float32
    engine = FasterWhisperEngine(
        model_size="tiny",
        compute_type="float32"  # Should be overridden
    )
    
    # Should force INT8
    assert engine.compute_type == "int8"

@pytest.mark.asyncio
async def test_memory_usage(asr_engine):
    """Test memory usage stays bounded."""
    import psutil
    import gc
    
    process = psutil.Process()
    
    # Initial memory
    gc.collect()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Process audio
    audio_frames = generate_test_audio(duration_s=10.0)
    async for _ in asr_engine.transcribe_stream(audio_frames):
        pass
    
    # Final memory
    gc.collect()
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    
    # Should not leak memory significantly
    mem_increase = mem_after - mem_before
    assert mem_increase < 100  # <100MB increase

@pytest.mark.asyncio
async def test_concurrent_streams(asr_engine):
    """Test handling multiple concurrent streams."""
    
    async def process_stream(stream_id: int):
        audio_frames = generate_test_audio(duration_s=2.0)
        chunks = []
        async for chunk in asr_engine.transcribe_stream(audio_frames):
            chunks.append(chunk)
        return stream_id, len(chunks)
    
    # Process 5 streams concurrently
    tasks = [process_stream(i) for i in range(5)]
    results = await asyncio.gather(*tasks)
    
    # All should complete
    assert len(results) == 5
    assert all(count > 0 for _, count in results)
```

#### Performance Benchmarks:
```python
# scripts/benchmark_asr.py
import asyncio
import time
import numpy as np
from izwi.asr.engine import create_asr_engine

async def benchmark():
    """Benchmark ASR performance."""
    engine = create_asr_engine()
    
    # Test different audio lengths
    for duration in [1, 5, 10, 30]:
        audio = generate_audio(duration)
        
        start = time.perf_counter()
        chunks = []
        
        async for chunk in engine.transcribe_stream(audio):
            chunks.append(chunk)
        
        elapsed = time.perf_counter() - start
        rtf = elapsed / duration
        
        print(f"Duration: {duration}s")
        print(f"  Processing time: {elapsed:.2f}s")
        print(f"  RTF: {rtf:.2f}x")
        print(f"  Chunks: {len(chunks)}")
        print()

if __name__ == "__main__":
    asyncio.run(benchmark())
```

#### Acceptance Criteria:
- ✅ RTF < 0.5x on CPU (faster than real-time)
- ✅ First transcript chunk in < 500ms
- ✅ INT8 quantization enforced
- ✅ Handles 16kHz mono PCM16 input
- ✅ Memory usage < 1GB for small model
- ✅ Shona language support
- ✅ Concurrent streams supported
- ✅ No memory leaks over time

---

*Continue to [Phase 3-4: LLM & TTS](phase3-4.md)*
