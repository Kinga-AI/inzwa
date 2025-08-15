# Phase 3-4: LLM & TTS Services

## Phase 3: LLM Service (Week 3)

### 3.1 LLM Engine Integration
**Owner:** ML Engineer  
**Duration:** 6 hours  
**Dependencies:** Phase 2 complete

#### Tasks:
- [ ] 3.1.1 Integrate llama-cpp-python with Q4/Q5
- [ ] 3.1.2 Add vLLM as optional flag
- [ ] 3.1.3 Implement streaming generation
- [ ] 3.1.4 Add context management
- [ ] 3.1.5 Implement token limits & load reduction

#### Code Implementation:
```python
# src/inzwa/llm/engine.py
from abc import ABC, abstractmethod
from typing import AsyncIterator, List, Dict, Optional
from llama_cpp import Llama
import asyncio
import time
import psutil

from inzwa.models import TokenChunk, Message
from inzwa.ops.settings import settings
from inzwa.ops.logging import logger
from inzwa.ops.metrics import METRICS

class LLMEngine(ABC):
    """Abstract base for LLM engines."""
    
    @abstractmethod
    async def generate_stream(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> AsyncIterator[TokenChunk]:
        """Stream token generation."""
        pass

class LlamaCppEngine(LLMEngine):
    """llama-cpp-python implementation (CPU-first)."""
    
    def __init__(
        self,
        model_path: str = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,  # CPU by default per .cursorrules
        n_threads: int = 4
    ):
        self.model_path = model_path or f"./models/{settings.llm_model}.gguf"
        
        # Validate quantization
        if "Q4" not in self.model_path and "Q5" not in self.model_path:
            logger.warning(
                f"Model {self.model_path} may not be quantized. "
                "Q4/Q5 required per .cursorrules"
            )
        
        logger.info(f"Loading LLM model: {self.model_path}")
        
        self.model = Llama(
            model_path=self.model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            use_mlock=True,  # Keep model in RAM
            verbose=False,
            seed=42  # Reproducible for testing
        )
        
        self.max_tokens_default = settings.llm_max_tokens
        self.context_window = n_ctx
        
        # Load reduction thresholds
        self.cpu_threshold = 80  # Reduce tokens if CPU > 80%
        self.memory_threshold = 85  # Reduce if memory > 85%
        
        logger.info("LLM model loaded successfully")
    
    async def generate_stream(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> AsyncIterator[TokenChunk]:
        """
        Generate tokens with streaming.
        
        Implements load reduction per .cursorrules:
        - Reduces max_tokens under high CPU/memory load
        - Tracks TTFB and tokens/sec metrics
        """
        # Format prompt
        prompt = self._format_prompt(messages)
        
        # Check system load and adjust tokens
        max_tokens = await self._get_adjusted_max_tokens(max_tokens)
        
        logger.debug(
            f"Generating with max_tokens={max_tokens}, "
            f"temperature={temperature}"
        )
        
        # Track timing
        start_time = time.perf_counter()
        first_token = True
        token_count = 0
        
        # Generate with streaming
        stream = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                stop=["</s>", "\n\nUser:", "\n\nHuman:"],
                echo=False,
                top_p=0.95,
                repeat_penalty=1.1
            )
        )
        
        # Stream tokens
        for output in stream:
            token = output["choices"][0]["text"]
            is_final = output["choices"][0]["finish_reason"] is not None
            logprob = output["choices"][0].get("logprobs")
            
            # Track TTFB
            if first_token:
                ttfb = (time.perf_counter() - start_time) * 1000
                METRICS.llm_ttfb.observe(ttfb)
                logger.debug(f"LLM TTFB: {ttfb:.0f}ms")
                first_token = False
            
            token_count += 1
            
            yield TokenChunk(
                token=token,
                logprob=logprob,
                is_final=is_final
            )
            
            if is_final:
                break
        
        # Track tokens/sec
        elapsed = time.perf_counter() - start_time
        if elapsed > 0:
            tokens_per_sec = token_count / elapsed
            METRICS.llm_tokens_per_sec.set(tokens_per_sec)
            logger.debug(
                f"Generated {token_count} tokens in {elapsed:.2f}s "
                f"({tokens_per_sec:.1f} tok/s)"
            )
    
    def _format_prompt(self, messages: List[Message]) -> str:
        """
        Format messages for Mistral/Llama format.
        
        Supports system, user, and assistant roles.
        """
        prompt_parts = []
        
        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}\n")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}\n")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}\n")
        
        # Add final assistant prompt
        prompt_parts.append("Assistant:")
        
        return "".join(prompt_parts)
    
    async def _get_adjusted_max_tokens(
        self,
        requested: Optional[int]
    ) -> int:
        """
        Adjust max tokens based on system load.
        
        Per .cursorrules: reduce max_new_tokens under load.
        """
        base_tokens = requested or self.max_tokens_default
        
        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Check memory usage
        memory_percent = psutil.virtual_memory().percent
        
        # Apply reductions
        if cpu_percent > self.cpu_threshold:
            reduction = (cpu_percent - self.cpu_threshold) / 100
            base_tokens = int(base_tokens * (1 - reduction))
            logger.warning(
                f"High CPU ({cpu_percent:.0f}%), "
                f"reducing tokens to {base_tokens}"
            )
        
        if memory_percent > self.memory_threshold:
            reduction = (memory_percent - self.memory_threshold) / 100
            base_tokens = int(base_tokens * (1 - reduction))
            logger.warning(
                f"High memory ({memory_percent:.0f}%), "
                f"reducing tokens to {base_tokens}"
            )
        
        # Enforce minimum
        return max(base_tokens, 50)

class VLLMEngine(LLMEngine):
    """vLLM implementation (GPU, optional flag)."""
    
    def __init__(self):
        if not settings.enable_vllm:
            raise ValueError(
                "vLLM disabled. Set INZWA_ENABLE_VLLM=true"
            )
        
        # TODO: Implement vLLM client
        logger.warning("vLLM engine not yet implemented")
    
    async def generate_stream(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> AsyncIterator[TokenChunk]:
        # Placeholder
        yield TokenChunk(
            token="[vLLM not implemented]",
            logprob=None,
            is_final=True
        )

# Factory function
def create_llm_engine() -> LLMEngine:
    """Create LLM engine based on settings."""
    if settings.llm_engine == "llama-cpp":
        return LlamaCppEngine()
    elif settings.llm_engine == "vllm":
        return VLLMEngine()
    else:
        raise ValueError(f"Unknown LLM engine: {settings.llm_engine}")
```

#### Testing:
```python
# tests/test_llm_engine.py
import pytest
import asyncio
import time
from unittest.mock import patch, MagicMock
from inzwa.llm.engine import LlamaCppEngine
from inzwa.models import Message

@pytest.fixture
async def llm_engine():
    """Create test LLM engine with small model."""
    # Use a tiny test model for CI
    engine = LlamaCppEngine(
        model_path="tests/models/tinyllama-128m-Q4.gguf",
        n_ctx=512,  # Smaller context for tests
        n_threads=2
    )
    yield engine

@pytest.mark.asyncio
async def test_llm_streaming(llm_engine):
    """Test streaming token generation."""
    messages = [
        Message(role="user", content="Hello, how are you?")
    ]
    
    tokens = []
    start = time.perf_counter()
    first_token_time = None
    
    async for chunk in llm_engine.generate_stream(messages, max_tokens=50):
        tokens.append(chunk.token)
        
        # Track TTFB
        if len(tokens) == 1:
            first_token_time = time.perf_counter() - start
    
    # Check TTFB
    assert first_token_time is not None
    assert first_token_time < 0.9  # <900ms per .cursorrules
    
    # Check we got tokens
    assert len(tokens) > 0
    assert len(tokens) <= 50  # Respects max_tokens
    
    # Check tokens/sec
    elapsed = time.perf_counter() - start
    tokens_per_sec = len(tokens) / elapsed
    assert tokens_per_sec >= 10  # >=10 tok/s per .cursorrules

@pytest.mark.asyncio
async def test_llm_load_reduction(llm_engine):
    """Test token reduction under load."""
    messages = [
        Message(role="user", content="Tell me a story")
    ]
    
    # Mock high CPU load
    with patch("psutil.cpu_percent", return_value=90.0):
        tokens = []
        async for chunk in llm_engine.generate_stream(
            messages,
            max_tokens=500
        ):
            tokens.append(chunk.token)
        
        # Should reduce tokens under load
        # (90% CPU = 10% over threshold = 10% reduction)
        assert len(tokens) < 500

@pytest.mark.asyncio
async def test_llm_memory_reduction(llm_engine):
    """Test token reduction under memory pressure."""
    messages = [
        Message(role="user", content="Explain quantum physics")
    ]
    
    # Mock high memory usage
    mock_memory = MagicMock()
    mock_memory.percent = 95.0  # 95% memory used
    
    with patch("psutil.virtual_memory", return_value=mock_memory):
        tokens = []
        async for chunk in llm_engine.generate_stream(
            messages,
            max_tokens=500
        ):
            tokens.append(chunk.token)
        
        # Should significantly reduce tokens
        assert len(tokens) < 300

def test_llm_quantization_check():
    """Verify quantization is enforced."""
    # Should warn about non-quantized model
    with patch("inzwa.ops.logging.logger.warning") as mock_warn:
        engine = LlamaCppEngine(
            model_path="models/unquantized.gguf"
        )
        
        # Should have warned
        mock_warn.assert_called()
        assert "Q4/Q5 required" in str(mock_warn.call_args)

@pytest.mark.asyncio
async def test_llm_prompt_formatting(llm_engine):
    """Test prompt formatting for different roles."""
    messages = [
        Message(role="system", content="You are a helpful assistant"),
        Message(role="user", content="What's the weather?"),
        Message(role="assistant", content="I'll check that for you"),
        Message(role="user", content="Thanks")
    ]
    
    prompt = llm_engine._format_prompt(messages)
    
    # Check format
    assert "System: You are a helpful assistant" in prompt
    assert "User: What's the weather?" in prompt
    assert "Assistant: I'll check that for you" in prompt
    assert "User: Thanks" in prompt
    assert prompt.endswith("Assistant:")

@pytest.mark.asyncio
async def test_llm_concurrent_generation(llm_engine):
    """Test handling concurrent generation requests."""
    
    async def generate(prompt: str):
        messages = [Message(role="user", content=prompt)]
        tokens = []
        async for chunk in llm_engine.generate_stream(
            messages,
            max_tokens=20
        ):
            tokens.append(chunk.token)
        return "".join(tokens)
    
    # Generate concurrently
    prompts = ["Hello", "Hi there", "Good morning"]
    results = await asyncio.gather(*[generate(p) for p in prompts])
    
    # All should complete
    assert len(results) == 3
    assert all(len(r) > 0 for r in results)

@pytest.mark.asyncio
async def test_llm_memory_usage(llm_engine):
    """Test memory usage stays bounded."""
    import gc
    import psutil
    
    process = psutil.Process()
    
    # Initial memory
    gc.collect()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Generate multiple times
    for _ in range(10):
        messages = [Message(role="user", content="Hello")]
        async for _ in llm_engine.generate_stream(messages, max_tokens=20):
            pass
    
    # Final memory
    gc.collect()
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    
    # Should not leak significantly
    mem_increase = mem_after - mem_before
    assert mem_increase < 50  # <50MB increase
```

#### Performance Benchmarks:
```python
# scripts/benchmark_llm.py
import asyncio
import time
from inzwa.llm.engine import create_llm_engine
from inzwa.models import Message

async def benchmark():
    """Benchmark LLM performance."""
    engine = create_llm_engine()
    
    test_prompts = [
        "Hello",
        "What is the capital of Zimbabwe?",
        "Tell me a short story about a lion",
        "Explain photosynthesis in simple terms"
    ]
    
    print("LLM Performance Benchmarks")
    print("=" * 50)
    
    for prompt in test_prompts:
        messages = [Message(role="user", content=prompt)]
        
        tokens = []
        start = time.perf_counter()
        ttfb = None
        
        async for chunk in engine.generate_stream(messages):
            if len(tokens) == 0:
                ttfb = (time.perf_counter() - start) * 1000
            tokens.append(chunk.token)
        
        elapsed = time.perf_counter() - start
        
        print(f"\nPrompt: {prompt[:50]}...")
        print(f"  TTFB: {ttfb:.0f}ms")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Tokens: {len(tokens)}")
        print(f"  Tokens/sec: {len(tokens)/elapsed:.1f}")
        print(f"  Response: {''.join(tokens[:100])}...")

if __name__ == "__main__":
    asyncio.run(benchmark())
```

#### Acceptance Criteria:
- ✅ TTFB ≤ 900ms on CPU
- ✅ ≥10 tokens/s on CPU
- ✅ Q4/Q5 quantization enforced
- ✅ Streaming works smoothly
- ✅ Reduces tokens under CPU load (>80%)
- ✅ Reduces tokens under memory pressure (>85%)
- ✅ Memory usage < 4GB for 2B model
- ✅ Handles concurrent requests
- ✅ No memory leaks

---

### 3.2 Safety Filters
**Owner:** ML Engineer  
**Duration:** 3 hours  
**Dependencies:** 3.1

#### Tasks:
- [ ] 3.2.1 Implement input sanitization
- [ ] 3.2.2 Add output content filtering
- [ ] 3.2.3 Create refusal templates
- [ ] 3.2.4 Add topic boundaries
- [ ] 3.2.5 Log safety violations (HMAC only)

#### Code Implementation:
```python
# src/inzwa/llm/safety.py
import re
import hashlib
import hmac
from typing import Optional, List
from dataclasses import dataclass

from inzwa.ops.settings import settings
from inzwa.ops.logging import logger

@dataclass
class SafetyResult:
    """Result of safety check."""
    is_safe: bool
    reason: Optional[str] = None
    filtered_text: Optional[str] = None

class SafetyFilter:
    """
    Simple safety filter per .cursorrules.
    
    Uses regex patterns only - no heavy frameworks.
    """
    
    def __init__(self):
        # Harmful content patterns
        self.harmful_input_patterns = [
            # Violence
            (r'\b(kill|murder|hurt|harm|attack|destroy)\s+\w+', "violence"),
            # Hate speech
            (r'\b(hate|racist|sexist|discriminat)\w*\b', "hate_speech"),
            # Illegal activities
            (r'\b(hack|crack|steal|illegal|crime)\b', "illegal"),
            # Self-harm
            (r'\b(suicide|self.?harm|cut\s+myself)\b', "self_harm"),
        ]
        
        # PII patterns for output filtering
        self.pii_patterns = [
            (r'\b\d{3}-\d{2}-\d{4}\b', "[SSN]"),  # SSN
            (r'\b\d{16}\b', "[CARD]"),  # Credit card
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "[EMAIL]"),
            (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', "[PHONE]"),
        ]
        
        # Safe topics (whitelist)
        self.safe_topics = [
            "greeting", "weather", "time", "date",
            "education", "culture", "language", "history",
            "science", "technology", "health", "food",
            "travel", "music", "art", "sports"
        ]
        
        # Refusal templates
        self.refusal_templates = [
            "I cannot help with that request. Please ask something else.",
            "That topic is outside my boundaries. How else can I assist you?",
            "I'm not able to discuss that. Let's talk about something else.",
        ]
        
        # HMAC key for logging
        self.hmac_key = settings.hmac_key.encode() if hasattr(settings, 'hmac_key') else b"default-key"
    
    def check_input(self, text: str) -> SafetyResult:
        """
        Check input for harmful content.
        
        Returns SafetyResult with refusal message if unsafe.
        """
        text_lower = text.lower()
        
        # Check harmful patterns
        for pattern, category in self.harmful_input_patterns:
            if re.search(pattern, text_lower):
                # Log with HMAC (no raw content)
                user_hash = self._get_user_hash(text[:50])
                logger.warning(
                    "Safety filter triggered",
                    category=category,
                    user_hash=user_hash
                )
                
                # Return refusal
                import random
                refusal = random.choice(self.refusal_templates)
                
                return SafetyResult(
                    is_safe=False,
                    reason=f"Detected {category}",
                    filtered_text=refusal
                )
        
        return SafetyResult(is_safe=True)
    
    def check_output(self, text: str) -> SafetyResult:
        """
        Filter output for PII and inappropriate content.
        
        Returns SafetyResult with filtered text.
        """
        filtered = text
        
        # Remove PII
        for pattern, replacement in self.pii_patterns:
            if re.search(pattern, filtered):
                logger.warning(
                    "PII detected in output",
                    type=replacement,
                    user_hash=self._get_user_hash(text[:50])
                )
                filtered = re.sub(pattern, replacement, filtered)
        
        # Check if output contains harmful content
        for pattern, category in self.harmful_input_patterns:
            if re.search(pattern, filtered.lower()):
                logger.warning(
                    "Harmful content in output",
                    category=category,
                    user_hash=self._get_user_hash(text[:50])
                )
                # Replace with generic response
                return SafetyResult(
                    is_safe=False,
                    reason=f"Output contains {category}",
                    filtered_text="I cannot provide that information."
                )
        
        return SafetyResult(
            is_safe=True,
            filtered_text=filtered
        )
    
    def is_safe_topic(self, topic: str) -> bool:
        """
        Check if topic is within safe boundaries.
        
        Used for topic-based filtering.
        """
        topic_lower = topic.lower()
        
        # Check if any safe topic keyword is present
        for safe_topic in self.safe_topics:
            if safe_topic in topic_lower:
                return True
        
        # Check for explicitly unsafe topics
        unsafe_keywords = ["adult", "nsfw", "explicit", "violent"]
        for unsafe in unsafe_keywords:
            if unsafe in topic_lower:
                return False
        
        # Default to safe for unknown topics
        return True
    
    def _get_user_hash(self, text: str) -> str:
        """
        Generate HMAC hash for logging.
        
        Never logs raw user content.
        """
        h = hmac.new(self.hmac_key, text.encode(), hashlib.sha256)
        return h.hexdigest()[:16]  # First 16 chars

# Global instance
safety_filter = SafetyFilter()
```

#### Testing:
```python
# tests/test_safety.py
import pytest
from inzwa.llm.safety import SafetyFilter, SafetyResult

@pytest.fixture
def safety():
    return SafetyFilter()

def test_input_violence_detection(safety):
    """Test detection of violent content."""
    result = safety.check_input("How do I kill someone?")
    
    assert not result.is_safe
    assert "violence" in result.reason.lower()
    assert result.filtered_text  # Should have refusal
    assert "cannot" in result.filtered_text.lower()

def test_input_safe_content(safety):
    """Test safe content passes."""
    result = safety.check_input("What's the weather today?")
    
    assert result.is_safe
    assert result.reason is None
    assert result.filtered_text is None

def test_output_pii_filtering(safety):
    """Test PII removal from output."""
    text = "My SSN is 123-45-6789 and card is 1234567812345678"
    result = safety.check_output(text)
    
    assert result.is_safe
    assert "[SSN]" in result.filtered_text
    assert "[CARD]" in result.filtered_text
    assert "123-45-6789" not in result.filtered_text
    assert "1234567812345678" not in result.filtered_text

def test_output_email_filtering(safety):
    """Test email filtering."""
    text = "Contact me at john.doe@example.com"
    result = safety.check_output(text)
    
    assert "[EMAIL]" in result.filtered_text
    assert "@example.com" not in result.filtered_text

def test_safe_topics(safety):
    """Test topic boundary checking."""
    # Safe topics
    assert safety.is_safe_topic("weather forecast")
    assert safety.is_safe_topic("education resources")
    assert safety.is_safe_topic("cooking recipes")
    
    # Unsafe topics
    assert not safety.is_safe_topic("adult content")
    assert not safety.is_safe_topic("nsfw material")
    assert not safety.is_safe_topic("violent games")

def test_no_raw_content_logged(safety, caplog):
    """Test that raw content is never logged."""
    sensitive_text = "My password is secret123"
    
    # Trigger safety check
    safety.check_input(sensitive_text)
    
    # Check logs
    log_text = caplog.text
    
    # Should NOT contain raw content
    assert "secret123" not in log_text
    assert "password" not in log_text
    
    # Should contain HMAC hash
    assert "user_hash=" in log_text

def test_hmac_consistency(safety):
    """Test HMAC hashing is consistent."""
    text = "Test content"
    
    hash1 = safety._get_user_hash(text)
    hash2 = safety._get_user_hash(text)
    
    # Same input should give same hash
    assert hash1 == hash2
    
    # Different input should give different hash
    hash3 = safety._get_user_hash("Different content")
    assert hash1 != hash3

def test_refusal_variety(safety):
    """Test different refusal messages."""
    refusals = set()
    
    # Collect refusal messages
    for _ in range(10):
        result = safety.check_input("How to hack a system")
        refusals.add(result.filtered_text)
    
    # Should have some variety
    assert len(refusals) > 1

def test_performance(safety):
    """Test filter performance."""
    import time
    
    text = "This is a normal message " * 100  # Long text
    
    start = time.perf_counter()
    for _ in range(100):
        safety.check_input(text)
        safety.check_output(text)
    elapsed = time.perf_counter() - start
    
    # Should be fast
    avg_time = elapsed / 200  # 100 input + 100 output
    assert avg_time < 0.005  # <5ms per check
```

#### Integration Test:
```python
# tests/test_llm_with_safety.py
import pytest
from inzwa.llm.engine import create_llm_engine
from inzwa.llm.safety import safety_filter
from inzwa.models import Message

@pytest.mark.asyncio
async def test_llm_with_safety():
    """Test LLM with safety filters."""
    engine = create_llm_engine()
    
    # Test harmful input
    harmful_msg = Message(role="user", content="How to hurt someone")
    
    # Check input safety
    safety_result = safety_filter.check_input(harmful_msg.content)
    
    if not safety_result.is_safe:
        # Return refusal instead of generating
        response = safety_result.filtered_text
    else:
        # Generate normally
        tokens = []
        async for chunk in engine.generate_stream([harmful_msg]):
            tokens.append(chunk.token)
        response = "".join(tokens)
        
        # Filter output
        output_result = safety_filter.check_output(response)
        response = output_result.filtered_text
    
    # Should have refused
    assert "cannot" in response.lower() or "not able" in response.lower()
```

#### Acceptance Criteria:
- ✅ Blocks harmful inputs (violence, hate, illegal)
- ✅ Redacts PII in outputs (SSN, cards, emails, phones)
- ✅ Uses refusal templates for unsafe requests
- ✅ Topic boundaries enforced
- ✅ No raw content in logs (HMAC hashes only)
- ✅ <5ms processing time per check
- ✅ Multiple refusal templates for variety
- ✅ Consistent HMAC hashing

---

## Phase 4: TTS Service (Week 4)

### 4.1 TTS Engine Integration
**Owner:** ML Engineer  
**Duration:** 6 hours  
**Dependencies:** Phase 3 complete

#### Tasks:
- [ ] 4.1.1 Integrate Coqui TTS VITS-lite
- [ ] 4.1.2 Export model to ONNX
- [ ] 4.1.3 Implement streaming synthesis
- [ ] 4.1.4 Add voice selection
- [ ] 4.1.5 Implement caching

#### Code Implementation:
```python
# src/inzwa/tts/engine.py
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional, Dict
import numpy as np
import hashlib
import asyncio
import time
from collections import OrderedDict

from inzwa.models import AudioChunk
from inzwa.ops.settings import settings
from inzwa.ops.logging import logger
from inzwa.ops.metrics import METRICS

class TTSEngine(ABC):
    """Abstract base for TTS engines."""
    
    @abstractmethod
    async def synthesize_stream(
        self,
        text: str,
        voice: str = "default"
    ) -> AsyncIterator[AudioChunk]:
        """Stream audio synthesis."""
        pass

class CoquiTTSEngine(TTSEngine):
    """Coqui TTS implementation with ONNX support."""
    
    def __init__(
        self,
        model_name: str = None,
        use_onnx: bool = True,
        cache_size: int = 100
    ):
        self.model_name = model_name or settings.tts_model
        self.use_onnx = use_onnx and settings.tts_use_onnx
        
        # Initialize cache (LRU)
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.cache_size = cache_size
        
        if self.use_onnx:
            self._init_onnx()
        else:
            self._init_native()
        
        self.sample_rate = 22050  # VITS default
        self.output_sample_rate = 16000  # Output at 16kHz
        
        logger.info(
            f"TTS engine initialized "
            f"({'ONNX' if self.use_onnx else 'Native'})"
        )
    
    def _init_onnx(self):
        """Initialize ONNX runtime."""
        import onnxruntime as ort
        
        model_path = f"./models/{self.model_name}.onnx"
        logger.info(f"Loading ONNX model: {model_path}")
        
        # Create session with optimization
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options,
            providers=['CPUExecutionProvider']
        )
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
    
    def _init_native(self):
        """Initialize native Coqui TTS."""
        from TTS.api import TTS
        
        logger.info(f"Loading native TTS model: {self.model_name}")
        self.tts = TTS(self.model_name)
    
    async def synthesize_stream(
        self,
        text: str,
        voice: str = "default"
    ) -> AsyncIterator[AudioChunk]:
        """
        Stream audio synthesis with caching.
        
        Implements:
        - Idempotent caching per .cursorrules
        - Streaming in 20-40ms chunks
        - TTFW < 300ms target
        """
        # Check cache
        cache_key = self._get_cache_key(text, voice)
        
        if cache_key in self.cache:
            # Move to end (LRU)
            self.cache.move_to_end(cache_key)
            
            # Stream from cache
            audio = self.cache[cache_key]
            logger.debug(f"TTS cache hit for: '{text[:30]}...'")
            
            async for chunk in self._stream_audio(audio, from_cache=True):
                yield chunk
            return
        
        # Synthesize
        start_time = time.perf_counter()
        
        if self.use_onnx:
            audio = await self._synthesize_onnx(text)
        else:
            audio = await self._synthesize_native(text)
        
        # Track synthesis time
        synthesis_time = time.perf_counter() - start_time
        logger.debug(f"TTS synthesis took {synthesis_time:.2f}s")
        
        # Cache if small enough
        if len(audio) < self.output_sample_rate * 10:  # <10s
            self._add_to_cache(cache_key, audio)
        
        # Stream chunks
        first_chunk = True
        async for chunk in self._stream_audio(audio):
            if first_chunk:
                ttfw = (time.perf_counter() - start_time) * 1000
                METRICS.tts_ttfw.observe(ttfw)
                logger.debug(f"TTS TTFW: {ttfw:.0f}ms")
                first_chunk = False
            
            yield chunk
    
    async def _synthesize_onnx(self, text: str) -> np.ndarray:
        """Synthesize using ONNX."""
        # Encode text
        text_encoded = self._encode_text(text)
        
        # Run inference in executor
        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(
            None,
            lambda: self.session.run(
                [self.output_name],
                {self.input_name: text_encoded}
            )
        )
        
        audio = outputs[0].squeeze()
        
        # Resample to 16kHz
        audio_16k = self._resample(audio, self.sample_rate, self.output_sample_rate)
        
        return audio_16k
    
    async def _synthesize_native(self, text: str) -> np.ndarray:
        """Synthesize using native TTS."""
        loop = asyncio.get_event_loop()
        
        # Generate in executor
        audio = await loop.run_in_executor(
            None,
            self.tts.tts,
            text
        )
        
        # Convert to numpy and resample
        audio = np.array(audio)
        audio_16k = self._resample(audio, self.sample_rate, self.output_sample_rate)
        
        return audio_16k
    
    async def _stream_audio(
        self,
        audio: np.ndarray,
        from_cache: bool = False
    ) -> AsyncIterator[AudioChunk]:
        """
        Stream audio in chunks.
        
        Chunks are 20-40ms per .cursorrules.
        """
        # Calculate chunk size (30ms default)
        chunk_duration_ms = 30
        chunk_size = int(self.output_sample_rate * chunk_duration_ms / 1000)
        
        # Convert to int16
        audio_int16 = (audio * 32767).astype(np.int16)
        
        for i in range(0, len(audio_int16), chunk_size):
            chunk = audio_int16[i:i + chunk_size]
            
            if len(chunk) == 0:
                break
            
            # Pad if too small
            if len(chunk) < chunk_size // 2:
                continue
            
            # Convert to bytes
            chunk_bytes = chunk.tobytes()
            
            # Optionally encode to Opus
            if settings.audio_out_codec == "opus":
                chunk_bytes = await self._encode_opus(chunk_bytes)
                format = "opus"
            else:
                format = "pcm16"
            
            yield AudioChunk(
                format=format,
                sample_rate=self.output_sample_rate,
                payload=chunk_bytes
            )
            
            # Small delay for streaming effect (skip if from cache)
            if not from_cache:
                await asyncio.sleep(0.001)
    
    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text for model input."""
        # Simple character encoding for demo
        # Real implementation would use proper tokenizer
        chars = list(text.lower())
        char_to_id = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz .,!?'-")}
        
        ids = [char_to_id.get(c, 0) for c in chars]
        return np.array(ids, dtype=np.int64).reshape(1, -1)
    
    def _resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """Resample audio to target sample rate."""
        if orig_sr == target_sr:
            return audio
        
        # Simple linear interpolation resampling
        duration = len(audio) / orig_sr
        target_length = int(duration * target_sr)
        
        indices = np.linspace(0, len(audio) - 1, target_length)
        resampled = np.interp(indices, np.arange(len(audio)), audio)
        
        return resampled
    
    async def _encode_opus(self, pcm_bytes: bytes) -> bytes:
        """Encode PCM to Opus."""
        # TODO: Implement Opus encoding
        # For now, return PCM
        return pcm_bytes
    
    def _get_cache_key(self, text: str, voice: str) -> str:
        """Generate cache key."""
        content = f"{text}:{voice}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _add_to_cache(self, key: str, audio: np.ndarray):
        """Add to LRU cache."""
        # Remove oldest if full
        if len(self.cache) >= self.cache_size:
            self.cache.popitem(last=False)
        
        self.cache[key] = audio
        logger.debug(f"Added to TTS cache (size: {len(self.cache)})")

# Factory function
def create_tts_engine() -> TTSEngine:
    """Create TTS engine based on settings."""
    if settings.tts_engine == "coqui-vits-lite":
        return CoquiTTSEngine()
    else:
        raise ValueError(f"Unknown TTS engine: {settings.tts_engine}")
```

#### Testing:
```python
# tests/test_tts_engine.py
import pytest
import asyncio
import time
import numpy as np
from inzwa.tts.engine import CoquiTTSEngine

@pytest.fixture
async def tts_engine():
    """Create test TTS engine."""
    # Use mock or tiny model for tests
    engine = CoquiTTSEngine(
        model_name="test_vits_tiny",
        use_onnx=True,
        cache_size=10
    )
    yield engine

@pytest.mark.asyncio
async def test_tts_streaming(tts_engine):
    """Test streaming synthesis."""
    text = "Hello, this is a test"
    
    chunks = []
    start = time.perf_counter()
    first_chunk_time = None
    
    async for chunk in tts_engine.synthesize_stream(text):
        chunks.append(chunk)
        
        if len(chunks) == 1:
            first_chunk_time = time.perf_counter() - start
    
    # Check TTFW
    assert first_chunk_time is not None
    assert first_chunk_time < 0.3  # <300ms per .cursorrules
    
    # Check we got audio
    assert len(chunks) > 0
    
    # Check chunk format
    for chunk in chunks:
        assert chunk.format in ["pcm16", "opus"]
        assert chunk.sample_rate == 16000
        assert len(chunk.payload) > 0
        
        # Check chunk size (20-40ms)
        if chunk.format == "pcm16":
            chunk_duration_ms = len(chunk.payload) / 2 / 16000 * 1000
            assert 15 <= chunk_duration_ms <= 45  # Some tolerance

@pytest.mark.asyncio
async def test_tts_caching(tts_engine):
    """Test idempotent caching."""
    text = "This should be cached"
    
    # First synthesis
    chunks1 = []
    start1 = time.perf_counter()
    async for chunk in tts_engine.synthesize_stream(text):
        chunks1.append(chunk)
    time1 = time.perf_counter() - start1
    
    # Second synthesis (should be cached)
    chunks2 = []
    start2 = time.perf_counter()
    async for chunk in tts_engine.synthesize_stream(text):
        chunks2.append(chunk)
    time2 = time.perf_counter() - start2
    
    # Cache should be much faster
    assert time2 < time1 * 0.1  # At least 10x faster
    
    # Should produce identical audio
    assert len(chunks1) == len(chunks2)
    for c1, c2 in zip(chunks1, chunks2):
        assert c1.payload == c2.payload

@pytest.mark.asyncio
async def test_tts_cache_lru(tts_engine):
    """Test LRU cache eviction."""
    # Fill cache beyond capacity
    for i in range(15):  # Cache size is 10
        text = f"Test message {i}"
        async for _ in tts_engine.synthesize_stream(text):
            pass
    
    # Cache should have last 10
    assert len(tts_engine.cache) == 10
    
    # First 5 should be evicted
    assert "Test message 0" not in str(tts_engine.cache.keys())
    assert "Test message 14" in str(tts_engine.cache.keys())

@pytest.mark.asyncio
async def test_tts_different_voices(tts_engine):
    """Test different voice selection."""
    text = "Hello world"
    
    # Different voices should have different cache keys
    key1 = tts_engine._get_cache_key(text, "voice1")
    key2 = tts_engine._get_cache_key(text, "voice2")
    
    assert key1 != key2

@pytest.mark.asyncio
async def test_tts_long_text(tts_engine):
    """Test synthesis of long text."""
    # Long text (>10s) should not be cached
    long_text = "This is a very long text. " * 100
    
    async for _ in tts_engine.synthesize_stream(long_text):
        pass
    
    # Should not be in cache
    cache_key = tts_engine._get_cache_key(long_text, "default")
    assert cache_key not in tts_engine.cache

@pytest.mark.asyncio
async def test_tts_concurrent_synthesis(tts_engine):
    """Test concurrent synthesis requests."""
    
    async def synthesize(text: str):
        chunks = []
        async for chunk in tts_engine.synthesize_stream(text):
            chunks.append(chunk)
        return len(chunks)
    
    # Synthesize concurrently
    texts = ["Hello", "World", "Testing"]
    results = await asyncio.gather(*[synthesize(t) for t in texts])
    
    # All should complete
    assert len(results) == 3
    assert all(r > 0 for r in results)

def test_tts_resampling(tts_engine):
    """Test audio resampling."""
    # Create test audio at 22050 Hz
    audio_22k = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050))
    
    # Resample to 16kHz
    audio_16k = tts_engine._resample(audio_22k, 22050, 16000)
    
    # Check length
    assert len(audio_16k) == 16000
    
    # Should preserve general shape
    assert np.corrcoef(audio_22k[:16000], audio_16k[:16000])[0, 1] > 0.9

@pytest.mark.asyncio
async def test_tts_memory_usage(tts_engine):
    """Test memory usage stays bounded."""
    import gc
    import psutil
    
    process = psutil.Process()
    
    # Initial memory
    gc.collect()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Synthesize multiple times
    for i in range(20):
        text = f"Test synthesis {i}"
        async for _ in tts_engine.synthesize_stream(text):
            pass
    
    # Final memory
    gc.collect()
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    
    # Should not leak significantly
    mem_increase = mem_after - mem_before
    assert mem_increase < 100  # <100MB increase
```

#### Performance Benchmarks:
```python
# scripts/benchmark_tts.py
import asyncio
import time
from inzwa.tts.engine import create_tts_engine

async def benchmark():
    """Benchmark TTS performance."""
    engine = create_tts_engine()
    
    test_texts = [
        "Hello",
        "How are you today?",
        "The weather is nice.",
        "This is a longer sentence to test the text to speech system performance."
    ]
    
    print("TTS Performance Benchmarks")
    print("=" * 50)
    
    for text in test_texts:
        print(f"\nText: {text}")
        
        # First run (no cache)
        chunks = []
        start = time.perf_counter()
        ttfw = None
        
        async for chunk in engine.synthesize_stream(text):
            if len(chunks) == 0:
                ttfw = (time.perf_counter() - start) * 1000
            chunks.append(chunk)
        
        time_nocache = time.perf_counter() - start
        
        print(f"  TTFW: {ttfw:.0f}ms")
        print(f"  Total time (no cache): {time_nocache:.2f}s")
        print(f"  Chunks: {len(chunks)}")
        
        # Second run (cached)
        start = time.perf_counter()
        async for _ in engine.synthesize_stream(text):
            pass
        time_cached = time.perf_counter() - start
        
        print(f"  Time (cached): {time_cached:.3f}s")
        print(f"  Speedup: {time_nocache/time_cached:.1f}x")

if __name__ == "__main__":
    asyncio.run(benchmark())
```

#### Acceptance Criteria:
- ✅ TTFW ≤ 300ms
- ✅ Streaming in 20-40ms chunks
- ✅ ONNX inference working
- ✅ Caching reduces latency 10x+
- ✅ LRU cache with configurable size
- ✅ Opus encoding optional
- ✅ Memory usage < 500MB
- ✅ Handles concurrent synthesis
- ✅ No memory leaks

---

*Continue to [Phase 5-6: Orchestration & WebSocket](phase5-6.md)*
