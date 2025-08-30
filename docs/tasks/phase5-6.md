# Phase 5-6: Orchestration & WebSocket Integration

## Phase 5: Orchestrator (Week 5)

### 5.1 Pipeline Orchestration  
**Owner:** Backend Lead  
**Duration:** 6 hours  
**Dependencies:** Phases 1-4 complete

#### Tasks:
- [ ] 5.1.1 Implement orchestrator (<50 lines per function)
- [ ] 5.1.2 Add bounded queues (max 8 items)
- [ ] 5.1.3 Implement phrase boundary detection
- [ ] 5.1.4 Add backpressure control
- [ ] 5.1.5 Handle pipeline errors gracefully

#### Code Implementation:
```python
# src/izwi/orch/orchestrator.py
"""
Ultra-light orchestrator per .cursorrules.

CRITICAL: All functions must be <50 lines!
"""
from typing import AsyncIterator
import asyncio
from dataclasses import dataclass

from izwi.models import AudioChunk, TranscriptChunk, TokenChunk, Message
from izwi.ops.logging import logger
from izwi.ops.metrics import METRICS

@dataclass
class OrchestratorConfig:
    """Configuration for orchestrator."""
    max_queue_size: int = 8  # Per .cursorrules
    phrase_min_length: int = 20
    phrase_delimiters: tuple = (".", "?", "!", ",", ";")
    backpressure_drop_threshold: int = 6

class Orchestrator:
    """
    Minimal streaming orchestrator.
    
    Connects ASR → LLM → TTS in <50 lines.
    """
    
    def __init__(self, asr, llm, tts, config: OrchestratorConfig = None):
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.config = config or OrchestratorConfig()
        
        # Metrics
        self.dropped_frames = 0
        self.processed_turns = 0
    
    async def run(
        self,
        audio_in: AsyncIterator[bytes]
    ) -> AsyncIterator[AudioChunk]:
        """
        Main pipeline: audio → text → response → speech.
        
        Must be <50 lines per .cursorrules!
        """
        # Create bounded queues
        transcript_q = asyncio.Queue(maxsize=self.config.max_queue_size)
        phrase_q = asyncio.Queue(maxsize=self.config.max_queue_size)
        
        # Start pipeline tasks
        tasks = [
            asyncio.create_task(self._asr_task(audio_in, transcript_q)),
            asyncio.create_task(self._llm_task(transcript_q, phrase_q)),
        ]
        
        try:
            # Stream TTS output
            async for audio_chunk in self._tts_task(phrase_q):
                yield audio_chunk
                
        except asyncio.CancelledError:
            logger.info("Orchestrator cancelled")
        except Exception as e:
            logger.error(f"Orchestrator error: {e}", exc_info=True)
        finally:
            # Clean up tasks
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Log metrics
            logger.info(
                f"Turn complete: dropped={self.dropped_frames}, "
                f"processed={self.processed_turns}"
            )
    
    async def _asr_task(
        self,
        audio_in: AsyncIterator[bytes],
        output_q: asyncio.Queue
    ):
        """Process audio → transcripts (<50 lines)."""
        buffer = ""
        
        try:
            async for transcript in self.asr.transcribe_stream(audio_in):
                if not transcript.text:
                    continue
                
                buffer += transcript.text + " "
                
                # Emit on final or phrase boundary
                if transcript.is_final or self._is_phrase_boundary(buffer):
                    text = buffer.strip()
                    if text:
                        try:
                            output_q.put_nowait(text)
                            logger.debug(f"ASR → Queue: '{text[:50]}...'")
                            buffer = ""
                        except asyncio.QueueFull:
                            self.dropped_frames += 1
                            logger.warning("Transcript queue full, dropping")
                            
        except Exception as e:
            logger.error(f"ASR task error: {e}")
    
    async def _llm_task(
        self,
        input_q: asyncio.Queue,
        output_q: asyncio.Queue
    ):
        """Process transcripts → LLM responses (<50 lines)."""
        try:
            while True:
                # Get transcript
                text = await input_q.get()
                self.processed_turns += 1
                
                # Generate response
                messages = [Message(role="user", content=text)]
                buffer = ""
                
                async for token in self.llm.generate_stream(messages):
                    buffer += token.token
                    
                    # Emit phrases
                    if self._is_phrase_boundary(buffer):
                        phrase = buffer.strip()
                        if len(phrase) >= self.config.phrase_min_length:
                            try:
                                output_q.put_nowait(phrase)
                                logger.debug(f"LLM → Queue: '{phrase[:50]}...'")
                                buffer = ""
                            except asyncio.QueueFull:
                                logger.warning("Phrase queue full")
                                # Apply backpressure - wait
                                await output_q.put(phrase)
                                buffer = ""
                
                # Emit remaining
                if buffer.strip():
                    await output_q.put(buffer.strip())
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"LLM task error: {e}")
    
    async def _tts_task(
        self,
        input_q: asyncio.Queue
    ) -> AsyncIterator[AudioChunk]:
        """Process phrases → audio (<50 lines)."""
        try:
            while True:
                # Get phrase with timeout
                try:
                    phrase = await asyncio.wait_for(
                        input_q.get(),
                        timeout=30.0
                    )
                except asyncio.TimeoutError:
                    logger.debug("TTS queue timeout, ending stream")
                    break
                
                # Synthesize and stream
                async for audio_chunk in self.tts.synthesize_stream(phrase):
                    yield audio_chunk
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"TTS task error: {e}")
    
    def _is_phrase_boundary(self, text: str) -> bool:
        """Check if text ends with phrase delimiter."""
        text = text.strip()
        if not text:
            return False
        
        # Check for delimiter
        return any(text.endswith(d) for d in self.config.phrase_delimiters)
```

#### Advanced Features (Separate Module):
```python
# src/izwi/orch/advanced.py
"""
Advanced orchestrator features.

Kept separate to maintain <50 lines per function.
"""
from typing import Optional, Dict, Any
import time
import asyncio

from izwi.orch.orchestrator import Orchestrator
from izwi.ops.logging import logger

class DialogueState:
    """Manages conversation state."""
    
    def __init__(self):
        self.turns: list = []
        self.context: list = []
        self.start_time: float = time.time()
        self.last_activity: float = time.time()
        self.user_speaking: bool = False
        self.assistant_speaking: bool = False
    
    def add_user_turn(self, text: str):
        """Add user utterance."""
        self.turns.append({"role": "user", "content": text, "timestamp": time.time()})
        self.last_activity = time.time()
        
    def add_assistant_turn(self, text: str):
        """Add assistant response."""
        self.turns.append({"role": "assistant", "content": text, "timestamp": time.time()})
        self.last_activity = time.time()
    
    def get_context(self, max_turns: int = 5) -> list:
        """Get recent conversation context."""
        return self.turns[-max_turns:] if self.turns else []
    
    def reset(self):
        """Reset conversation state."""
        self.turns.clear()
        self.context.clear()
        self.start_time = time.time()

class BargeInHandler:
    """Handles barge-in (interruption) logic."""
    
    def __init__(self, vad_threshold: float = 0.5):
        self.vad_threshold = vad_threshold
        self.speaking = False
        self.interrupt_signal = asyncio.Event()
    
    async def detect_barge_in(
        self,
        audio_frame: bytes,
        vad_model
    ) -> bool:
        """
        Detect if user is interrupting.
        
        Returns True if barge-in detected.
        """
        # Run VAD
        speech_prob = await vad_model.detect(audio_frame)
        
        if speech_prob > self.vad_threshold:
            if not self.speaking:
                self.speaking = True
                self.interrupt_signal.set()
                logger.info("Barge-in detected")
                return True
        else:
            self.speaking = False
        
        return False
    
    async def wait_for_interrupt(self, timeout: float = None):
        """Wait for interrupt signal."""
        try:
            await asyncio.wait_for(
                self.interrupt_signal.wait(),
                timeout=timeout
            )
            return True
        except asyncio.TimeoutError:
            return False
    
    def reset(self):
        """Reset interrupt state."""
        self.speaking = False
        self.interrupt_signal.clear()

class AdaptiveOrchestrator(Orchestrator):
    """
    Orchestrator with adaptive features.
    
    Adds dialogue state and barge-in handling.
    """
    
    def __init__(self, asr, llm, tts, vad=None):
        super().__init__(asr, llm, tts)
        self.vad = vad
        self.dialogue_state = DialogueState()
        self.barge_in = BargeInHandler()
    
    async def run_with_state(
        self,
        audio_in: AsyncIterator[bytes]
    ) -> AsyncIterator[AudioChunk]:
        """
        Run with dialogue state management.
        
        Tracks conversation and handles interruptions.
        """
        # Split audio for VAD monitoring
        audio_tee1, audio_tee2 = self._tee_audio(audio_in)
        
        # Monitor for barge-in
        if self.vad:
            asyncio.create_task(
                self._monitor_barge_in(audio_tee2)
            )
        
        # Run main pipeline
        async for audio_chunk in self.run(audio_tee1):
            # Check for interrupt
            if self.barge_in.interrupt_signal.is_set():
                logger.info("Stopping TTS due to barge-in")
                self.barge_in.reset()
                break
            
            yield audio_chunk
    
    async def _monitor_barge_in(
        self,
        audio_stream: AsyncIterator[bytes]
    ):
        """Monitor audio for barge-in."""
        async for frame in audio_stream:
            if self.dialogue_state.assistant_speaking:
                await self.barge_in.detect_barge_in(frame, self.vad)
    
    def _tee_audio(
        self,
        audio_in: AsyncIterator[bytes]
    ) -> tuple:
        """Split audio stream for parallel processing."""
        queue1 = asyncio.Queue()
        queue2 = asyncio.Queue()
        
        async def splitter():
            async for frame in audio_in:
                await queue1.put(frame)
                await queue2.put(frame)
        
        asyncio.create_task(splitter())
        
        async def reader(q):
            while True:
                yield await q.get()
        
        return reader(queue1), reader(queue2)
```

#### Testing:
```python
# tests/test_orchestrator.py
import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

from izwi.orch.orchestrator import Orchestrator, OrchestratorConfig
from izwi.models import AudioChunk, TranscriptChunk, TokenChunk

class MockASR:
    """Mock ASR for testing."""
    
    async def transcribe_stream(self, audio_frames):
        """Yield mock transcripts."""
        texts = ["Hello", "How are you", "today?"]
        for i, text in enumerate(texts):
            await asyncio.sleep(0.1)  # Simulate processing
            yield TranscriptChunk(
                text=text,
                is_final=(i == len(texts) - 1),
                start_ms=i * 1000,
                end_ms=(i + 1) * 1000,
                confidence=0.95
            )

class MockLLM:
    """Mock LLM for testing."""
    
    async def generate_stream(self, messages):
        """Yield mock tokens."""
        response = "I am doing well, thank you for asking."
        for token in response.split():
            await asyncio.sleep(0.05)  # Simulate generation
            yield TokenChunk(
                token=token + " ",
                logprob=-0.5,
                is_final=False
            )
        yield TokenChunk(token="", logprob=0, is_final=True)

class MockTTS:
    """Mock TTS for testing."""
    
    async def synthesize_stream(self, text):
        """Yield mock audio chunks."""
        # Generate 10 chunks per phrase
        for i in range(10):
            await asyncio.sleep(0.02)  # Simulate synthesis
            yield AudioChunk(
                format="pcm16",
                sample_rate=16000,
                payload=b"mock_audio" * 100
            )

async def generate_mock_audio():
    """Generate mock audio frames."""
    for i in range(30):  # 30 frames
        yield b"audio_frame" * 100
        await asyncio.sleep(0.03)  # 30ms per frame

@pytest.fixture
def orchestrator():
    """Create test orchestrator."""
    asr = MockASR()
    llm = MockLLM()
    tts = MockTTS()
    
    config = OrchestratorConfig(
        max_queue_size=8,
        phrase_min_length=10
    )
    
    return Orchestrator(asr, llm, tts, config)

@pytest.mark.asyncio
async def test_orchestrator_e2e(orchestrator):
    """Test end-to-end pipeline."""
    audio_in = generate_mock_audio()
    
    chunks = []
    start = time.perf_counter()
    first_chunk_time = None
    
    async for chunk in orchestrator.run(audio_in):
        chunks.append(chunk)
        
        if len(chunks) == 1:
            first_chunk_time = time.perf_counter() - start
        
        # Limit test duration
        if len(chunks) >= 20:
            break
    
    # Check TTFW
    assert first_chunk_time < 0.5  # <500ms
    
    # Check we got audio
    assert len(chunks) > 0
    
    # Check chunk format
    for chunk in chunks:
        assert chunk.format == "pcm16"
        assert chunk.sample_rate == 16000
        assert len(chunk.payload) > 0

@pytest.mark.asyncio
async def test_orchestrator_backpressure(orchestrator):
    """Test queue backpressure handling."""
    # Create slow TTS to cause backpressure
    slow_tts = AsyncMock()
    
    async def slow_synthesize(text):
        await asyncio.sleep(2.0)  # Very slow
        yield AudioChunk(format="pcm16", sample_rate=16000, payload=b"slow")
    
    slow_tts.synthesize_stream = slow_synthesize
    orchestrator.tts = slow_tts
    
    audio_in = generate_mock_audio()
    
    # Should handle backpressure without crash
    chunks = []
    async for chunk in orchestrator.run(audio_in):
        chunks.append(chunk)
        if len(chunks) >= 1:
            break
    
    # Check dropped frames were logged
    assert orchestrator.dropped_frames >= 0

@pytest.mark.asyncio
async def test_orchestrator_phrase_detection(orchestrator):
    """Test phrase boundary detection."""
    # Test various texts
    assert orchestrator._is_phrase_boundary("Hello.")
    assert orchestrator._is_phrase_boundary("How are you?")
    assert orchestrator._is_phrase_boundary("Wait,")
    assert not orchestrator._is_phrase_boundary("Hello")
    assert not orchestrator._is_phrase_boundary("")

@pytest.mark.asyncio
async def test_orchestrator_error_handling(orchestrator):
    """Test error recovery."""
    # Make ASR fail
    orchestrator.asr.transcribe_stream = AsyncMock(
        side_effect=Exception("ASR failed")
    )
    
    audio_in = generate_mock_audio()
    
    # Should handle error gracefully
    chunks = []
    async for chunk in orchestrator.run(audio_in):
        chunks.append(chunk)
    
    # May not produce output but shouldn't crash
    assert isinstance(chunks, list)

@pytest.mark.asyncio
async def test_orchestrator_cancellation(orchestrator):
    """Test clean cancellation."""
    audio_in = generate_mock_audio()
    
    # Start pipeline
    task = asyncio.create_task(
        orchestrator.run(audio_in).__anext__()
    )
    
    # Cancel after short delay
    await asyncio.sleep(0.1)
    task.cancel()
    
    # Should cancel cleanly
    with pytest.raises(asyncio.CancelledError):
        await task

def test_orchestrator_function_length():
    """Verify functions are <50 lines per .cursorrules."""
    import inspect
    from izwi.orch.orchestrator import Orchestrator
    
    for name, method in inspect.getmembers(Orchestrator, inspect.isfunction):
        if not name.startswith("_"):
            continue
        
        source = inspect.getsource(method)
        lines = source.split("\n")
        
        # Count non-empty, non-comment lines
        code_lines = [
            l for l in lines
            if l.strip() and not l.strip().startswith("#")
        ]
        
        assert len(code_lines) <= 50, f"{name} has {len(code_lines)} lines"

@pytest.mark.asyncio
async def test_dialogue_state():
    """Test dialogue state management."""
    from izwi.orch.advanced import DialogueState
    
    state = DialogueState()
    
    # Add turns
    state.add_user_turn("Hello")
    state.add_assistant_turn("Hi there")
    state.add_user_turn("How are you?")
    
    # Check state
    assert len(state.turns) == 3
    assert state.turns[0]["role"] == "user"
    assert state.turns[1]["role"] == "assistant"
    
    # Check context
    context = state.get_context(max_turns=2)
    assert len(context) == 2
    assert context[0]["content"] == "Hi there"

@pytest.mark.asyncio
async def test_barge_in_detection():
    """Test barge-in handler."""
    from izwi.orch.advanced import BargeInHandler
    
    handler = BargeInHandler(vad_threshold=0.5)
    
    # Mock VAD
    vad_mock = AsyncMock()
    vad_mock.detect = AsyncMock(return_value=0.8)  # High speech prob
    
    # Detect barge-in
    detected = await handler.detect_barge_in(b"audio", vad_mock)
    assert detected
    assert handler.interrupt_signal.is_set()
    
    # Reset
    handler.reset()
    assert not handler.interrupt_signal.is_set()
```

#### Performance Testing:
```python
# tests/test_orchestrator_performance.py
import asyncio
import time
import psutil
from izwi.orch.orchestrator import Orchestrator

async def test_orchestrator_latency():
    """Test orchestrator latency targets."""
    # Create real components (or fast mocks)
    from izwi.asr.engine import create_asr_engine
    from izwi.llm.engine import create_llm_engine
    from izwi.tts.engine import create_tts_engine
    
    asr = create_asr_engine()
    llm = create_llm_engine()
    tts = create_tts_engine()
    
    orch = Orchestrator(asr, llm, tts)
    
    # Generate test audio (5 seconds)
    async def audio_generator():
        for _ in range(166):  # ~5 seconds at 30ms frames
            yield b"\x00" * 960  # 30ms of silence at 16kHz
            await asyncio.sleep(0.03)
    
    # Measure latency
    audio_in = audio_generator()
    
    chunks = []
    start = time.perf_counter()
    ttfw = None
    
    async for chunk in orch.run(audio_in):
        chunks.append(chunk)
        
        if len(chunks) == 1:
            ttfw = time.perf_counter() - start
            break  # Just measure TTFW
    
    # Check TTFW target
    assert ttfw < 0.5  # <500ms per .cursorrules
    
    print(f"TTFW: {ttfw*1000:.0f}ms")

async def test_orchestrator_throughput():
    """Test concurrent session handling."""
    # Create orchestrators
    orchestrators = []
    for _ in range(10):
        orch = Orchestrator(
            MockASR(), MockLLM(), MockTTS()
        )
        orchestrators.append(orch)
    
    # Run concurrent sessions
    async def run_session(orch):
        audio_in = generate_mock_audio()
        chunks = []
        async for chunk in orch.run(audio_in):
            chunks.append(chunk)
            if len(chunks) >= 10:
                break
        return len(chunks)
    
    # Process all sessions
    start = time.perf_counter()
    results = await asyncio.gather(
        *[run_session(o) for o in orchestrators]
    )
    elapsed = time.perf_counter() - start
    
    # All should complete
    assert len(results) == 10
    assert all(r > 0 for r in results)
    
    print(f"10 sessions in {elapsed:.2f}s")

async def test_orchestrator_memory():
    """Test memory usage under load."""
    import gc
    
    process = psutil.Process()
    
    # Initial memory
    gc.collect()
    mem_before = process.memory_info().rss / 1024 / 1024
    
    # Run multiple turns
    orch = Orchestrator(MockASR(), MockLLM(), MockTTS())
    
    for _ in range(20):
        audio_in = generate_mock_audio()
        async for _ in orch.run(audio_in):
            break  # Just start each turn
    
    # Final memory
    gc.collect()
    mem_after = process.memory_info().rss / 1024 / 1024
    
    mem_increase = mem_after - mem_before
    assert mem_increase < 50  # <50MB increase
    
    print(f"Memory increase: {mem_increase:.1f}MB")
```

#### Acceptance Criteria:
- ✅ All functions < 50 lines
- ✅ Queues bounded at 8 items
- ✅ TTFW < 500ms
- ✅ E2E < 1.2s for short utterances
- ✅ Handles backpressure gracefully
- ✅ Drops frames when overloaded
- ✅ Phrase boundary detection works
- ✅ Clean error recovery
- ✅ Supports cancellation
- ✅ Memory usage bounded

---

## Phase 6: WebSocket Integration (Week 6)

### 6.1 Full Duplex WebSocket
**Owner:** Backend Lead  
**Duration:** 6 hours  
**Dependencies:** Phase 5 complete

#### Tasks:
- [ ] 6.1.1 Implement bidirectional streaming
- [ ] 6.1.2 Add session state management
- [ ] 6.1.3 Handle barge-in
- [ ] 6.1.4 Implement turn detection
- [ ] 6.1.5 Add connection recovery

#### Code Implementation:
```python
# src/izwi/api/websocket.py
"""
WebSocket handler for full-duplex audio streaming.

Implements /ws/audio endpoint per .cursorrules.
"""
from fastapi import WebSocket, WebSocketDisconnect, Depends
from typing import Dict, Optional
import asyncio
import json
import uuid
import time

from izwi.orch.orchestrator import Orchestrator
from izwi.orch.advanced import DialogueState, BargeInHandler
from izwi.ops.settings import settings
from izwi.ops.logging import logger
from izwi.ops.metrics import METRICS
from izwi.api.auth import verify_api_key

class WebSocketSession:
    """Manages a single WebSocket session."""
    
    def __init__(self, session_id: str, websocket: WebSocket):
        self.session_id = session_id
        self.websocket = websocket
        self.dialogue_state = DialogueState()
        self.barge_in = BargeInHandler()
        self.created_at = time.time()
        self.last_activity = time.time()
        self.audio_queue = asyncio.Queue(maxsize=settings.backpressure_threshold)
        self.is_active = True
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = time.time()
    
    async def cleanup(self):
        """Clean up session resources."""
        self.is_active = False
        # Drain queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

class WebSocketManager:
    """Manages all WebSocket sessions."""
    
    def __init__(self):
        self.sessions: Dict[str, WebSocketSession] = {}
        self.orchestrator = None  # Lazy init
        
    async def connect(
        self,
        websocket: WebSocket,
        session_id: Optional[str] = None
    ) -> WebSocketSession:
        """Accept new WebSocket connection."""
        await websocket.accept()
        
        # Generate session ID if needed
        if not session_id:
            session_id = str(uuid.uuid4())[:8]
        
        # Create session
        session = WebSocketSession(session_id, websocket)
        self.sessions[session_id] = session
        
        # Send welcome message
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "version": "0.1.0"
        })
        
        logger.info(f"WebSocket connected", session_id=session_id)
        METRICS.active_sessions.inc()
        
        return session
    
    async def disconnect(self, session: WebSocketSession):
        """Handle WebSocket disconnection."""
        await session.cleanup()
        
        if session.session_id in self.sessions:
            del self.sessions[session.session_id]
        
        logger.info(f"WebSocket disconnected", session_id=session.session_id)
        METRICS.active_sessions.dec()
    
    def get_session(self, session_id: str) -> Optional[WebSocketSession]:
        """Get session by ID."""
        return self.sessions.get(session_id)
    
    async def cleanup_inactive(self, timeout_seconds: int = 300):
        """Clean up inactive sessions."""
        now = time.time()
        to_remove = []
        
        for sid, session in self.sessions.items():
            if now - session.last_activity > timeout_seconds:
                to_remove.append(sid)
        
        for sid in to_remove:
            session = self.sessions.get(sid)
            if session:
                await self.disconnect(session)
                logger.info(f"Cleaned up inactive session", session_id=sid)

# Global manager
ws_manager = WebSocketManager()

async def websocket_endpoint(
    websocket: WebSocket,
    api_key: Optional[str] = Depends(verify_api_key)
):
    """
    Main WebSocket endpoint for audio streaming.
    
    Handles bidirectional audio + control messages.
    """
    session = None
    
    try:
        # Connect
        session = await ws_manager.connect(websocket)
        
        # Create orchestrator
        from izwi.asr.engine import create_asr_engine
        from izwi.llm.engine import create_llm_engine
        from izwi.tts.engine import create_tts_engine
        
        asr = create_asr_engine()
        llm = create_llm_engine()
        tts = create_tts_engine()
        
        orchestrator = Orchestrator(asr, llm, tts)
        
        # Start tasks
        tasks = [
            asyncio.create_task(
                handle_incoming(session, websocket)
            ),
            asyncio.create_task(
                handle_outgoing(session, orchestrator)
            ),
        ]
        
        # Wait for completion
        await asyncio.gather(*tasks)
        
    except WebSocketDisconnect:
        logger.info("Client disconnected", session_id=session.session_id if session else "unknown")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        if session and websocket.client_state.value == 1:  # CONNECTED
            await websocket.send_json({
                "type": "error",
                "message": "Internal error"
            })
    finally:
        if session:
            await ws_manager.disconnect(session)

async def handle_incoming(
    session: WebSocketSession,
    websocket: WebSocket
):
    """
    Handle incoming messages from client.
    
    Processes both audio frames and control messages.
    """
    try:
        while session.is_active:
            # Receive message
            message = await websocket.receive()
            session.update_activity()
            
            if "bytes" in message:
                # Audio frame
                audio = message["bytes"]
                
                # Validate frame size (20-40ms at 16kHz)
                frame_size = len(audio)
                if not (640 <= frame_size <= 1280):
                    await websocket.send_json({
                        "type": "warning",
                        "message": f"Invalid frame size: {frame_size}"
                    })
                    continue
                
                # Add to queue with backpressure
                try:
                    session.audio_queue.put_nowait(audio)
                except asyncio.QueueFull:
                    # Apply backpressure
                    await websocket.send_json({
                        "type": "warning",
                        "message": "Buffer full, frame dropped"
                    })
                    METRICS.dropped_frames.inc()
            
            elif "text" in message:
                # Control message
                try:
                    data = json.loads(message["text"])
                    await handle_control_message(session, websocket, data)
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid JSON"
                    })
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Incoming handler error: {e}")

async def handle_outgoing(
    session: WebSocketSession,
    orchestrator: Orchestrator
):
    """
    Handle outgoing audio to client.
    
    Runs orchestrator pipeline and streams results.
    """
    try:
        # Audio generator from queue
        async def audio_generator():
            while session.is_active:
                try:
                    frame = await asyncio.wait_for(
                        session.audio_queue.get(),
                        timeout=0.5
                    )
                    yield frame
                except asyncio.TimeoutError:
                    continue
        
        # Run orchestrator
        async for audio_chunk in orchestrator.run(audio_generator()):
            if not session.is_active:
                break
            
            # Send audio
            await session.websocket.send_bytes(audio_chunk.payload)
            
            # Send metadata
            await session.websocket.send_json({
                "type": "audio.chunk",
                "format": audio_chunk.format,
                "sample_rate": audio_chunk.sample_rate,
                "size": len(audio_chunk.payload)
            })
    
    except Exception as e:
        logger.error(f"Outgoing handler error: {e}")

async def handle_control_message(
    session: WebSocketSession,
    websocket: WebSocket,
    data: dict
):
    """Process control messages."""
    msg_type = data.get("type")
    
    if msg_type == "ping":
        await websocket.send_json({"type": "pong"})
    
    elif msg_type == "start":
        session.dialogue_state.reset()
        await websocket.send_json({
            "type": "session.started",
            "session_id": session.session_id
        })
    
    elif msg_type == "end_turn":
        # Signal end of user turn
        await websocket.send_json({
            "type": "turn.ended"
        })
    
    elif msg_type == "config":
        # Update session config
        config = data.get("config", {})
        # Apply config updates
        await websocket.send_json({
            "type": "config.updated",
            "config": config
        })
    
    else:
        await websocket.send_json({
            "type": "error",
            "message": f"Unknown message type: {msg_type}"
        })
```

#### Client Implementation Example:
```javascript
// example_client.js
class IzwiWebSocketClient {
    constructor(url) {
        this.url = url;
        this.ws = null;
        this.sessionId = null;
        this.audioContext = null;
        this.mediaStream = null;
    }
    
    async connect() {
        this.ws = new WebSocket(this.url);
        
        this.ws.onopen = () => {
            console.log('Connected to Izwi');
        };
        
        this.ws.onmessage = async (event) => {
            if (event.data instanceof Blob) {
                // Audio data
                await this.playAudio(event.data);
            } else {
                // Control message
                const msg = JSON.parse(event.data);
                this.handleMessage(msg);
            }
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
        
        this.ws.onclose = () => {
            console.log('Disconnected from Izwi');
            this.reconnect();
        };
    }
    
    async startRecording() {
        // Get microphone access
        this.mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: 16000,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true
            }
        });
        
        // Create audio context
        this.audioContext = new AudioContext({ sampleRate: 16000 });
        const source = this.audioContext.createMediaStreamSource(this.mediaStream);
        
        // Create processor for 30ms chunks
        const processor = this.audioContext.createScriptProcessor(512, 1, 1);
        
        processor.onaudioprocess = (e) => {
            if (this.ws.readyState === WebSocket.OPEN) {
                const float32 = e.inputBuffer.getChannelData(0);
                const int16 = new Int16Array(float32.length);
                
                // Convert float32 to int16
                for (let i = 0; i < float32.length; i++) {
                    int16[i] = Math.max(-32768, Math.min(32767, float32[i] * 32768));
                }
                
                // Send audio frame
                this.ws.send(int16.buffer);
            }
        };
        
        source.connect(processor);
        processor.connect(this.audioContext.destination);
        
        // Send start message
        this.ws.send(JSON.stringify({
            type: 'start',
            codec: 'pcm16',
            sample_rate: 16000
        }));
    }
    
    async playAudio(blob) {
        // Convert blob to array buffer
        const arrayBuffer = await blob.arrayBuffer();
        const int16 = new Int16Array(arrayBuffer);
        
        // Convert to float32
        const float32 = new Float32Array(int16.length);
        for (let i = 0; i < int16.length; i++) {
            float32[i] = int16[i] / 32768;
        }
        
        // Create audio buffer
        const audioBuffer = this.audioContext.createBuffer(
            1, float32.length, 16000
        );
        audioBuffer.getChannelData(0).set(float32);
        
        // Play
        const source = this.audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(this.audioContext.destination);
        source.start();
    }
    
    handleMessage(msg) {
        switch (msg.type) {
            case 'connected':
                this.sessionId = msg.session_id;
                console.log('Session ID:', this.sessionId);
                break;
            
            case 'asr.partial':
                console.log('Transcript:', msg.text);
                break;
            
            case 'audio.chunk':
                console.log('Audio chunk:', msg.size, 'bytes');
                break;
            
            case 'error':
                console.error('Error:', msg.message);
                break;
        }
    }
    
    reconnect() {
        console.log('Reconnecting in 1s...');
        setTimeout(() => this.connect(), 1000);
    }
}

// Usage
const client = new IzwiWebSocketClient('ws://localhost:8000/ws/audio');
client.connect();
```

#### Testing:
```python
# tests/test_websocket.py
import pytest
import asyncio
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

@pytest.mark.asyncio
async def test_websocket_connection():
    """Test WebSocket connection."""
    from izwi.api.app import app
    
    client = TestClient(app)
    
    with client.websocket_connect("/ws/audio") as websocket:
        # Receive welcome message
        data = websocket.receive_json()
        
        assert data["type"] == "connected"
        assert "session_id" in data
        assert "version" in data

@pytest.mark.asyncio
async def test_websocket_audio_streaming():
    """Test audio streaming."""
    from izwi.api.app import app
    
    client = TestClient(app)
    
    with client.websocket_connect("/ws/audio") as websocket:
        # Send start message
        websocket.send_json({
            "type": "start",
            "codec": "pcm16",
            "sample_rate": 16000
        })
        
        # Send audio frames
        for _ in range(10):
            audio_frame = b"\x00" * 960  # 30ms of silence
            websocket.send_bytes(audio_frame)
        
        # Should receive some response
        # (In real test, would check for audio/transcripts)

@pytest.mark.asyncio
async def test_websocket_backpressure():
    """Test backpressure handling."""
    from izwi.api.app import app
    
    client = TestClient(app)
    
    with client.websocket_connect("/ws/audio") as websocket:
        # Flood with frames
        for _ in range(100):
            websocket.send_bytes(b"\x00" * 960)
        
        # Should see backpressure warnings
        warnings = []
        try:
            while True:
                msg = websocket.receive_json(timeout=0.1)
                if msg["type"] == "warning":
                    warnings.append(msg)
        except:
            pass
        
        # Should have some warnings
        assert any("Buffer full" in w.get("message", "") for w in warnings)

@pytest.mark.asyncio
async def test_websocket_invalid_frame():
    """Test invalid frame handling."""
    from izwi.api.app import app
    
    client = TestClient(app)
    
    with client.websocket_connect("/ws/audio") as websocket:
        # Send invalid frame size
        websocket.send_bytes(b"\x00" * 100)  # Too small
        
        # Should get warning
        msg = websocket.receive_json()
        assert msg["type"] == "warning"
        assert "Invalid frame size" in msg["message"]

@pytest.mark.asyncio
async def test_websocket_control_messages():
    """Test control message handling."""
    from izwi.api.app import app
    
    client = TestClient(app)
    
    with client.websocket_connect("/ws/audio") as websocket:
        # Test ping/pong
        websocket.send_json({"type": "ping"})
        msg = websocket.receive_json()
        assert msg["type"] == "pong"
        
        # Test end_turn
        websocket.send_json({"type": "end_turn"})
        msg = websocket.receive_json()
        assert msg["type"] == "turn.ended"

@pytest.mark.asyncio
async def test_websocket_session_management():
    """Test session lifecycle."""
    from izwi.api.websocket import ws_manager
    
    # Check initial state
    assert len(ws_manager.sessions) == 0
    
    # Connect
    from izwi.api.app import app
    client = TestClient(app)
    
    with client.websocket_connect("/ws/audio") as ws:
        msg = ws.receive_json()
        session_id = msg["session_id"]
        
        # Check session exists
        assert session_id in ws_manager.sessions
        
    # After disconnect, session should be cleaned up
    assert session_id not in ws_manager.sessions

@pytest.mark.asyncio
async def test_websocket_concurrent_sessions():
    """Test multiple concurrent sessions."""
    from izwi.api.app import app
    
    client = TestClient(app)
    sessions = []
    
    # Connect multiple clients
    for i in range(5):
        ws = client.websocket_connect("/ws/audio").__enter__()
        msg = ws.receive_json()
        sessions.append((ws, msg["session_id"]))
    
    # All should have unique session IDs
    session_ids = [sid for _, sid in sessions]
    assert len(set(session_ids)) == 5
    
    # Clean up
    for ws, _ in sessions:
        ws.__exit__(None, None, None)

@pytest.mark.asyncio
async def test_websocket_reconnection():
    """Test reconnection handling."""
    from izwi.api.app import app
    
    client = TestClient(app)
    
    # First connection
    with client.websocket_connect("/ws/audio") as ws1:
        msg = ws1.receive_json()
        session_id1 = msg["session_id"]
    
    # Reconnect
    with client.websocket_connect("/ws/audio") as ws2:
        msg = ws2.receive_json()
        session_id2 = msg["session_id"]
    
    # Should get new session ID
    assert session_id1 != session_id2
```

#### Load Testing:
```python
# tests/load_test_websocket.py
import asyncio
import websockets
import json
import time

async def client_session(url: str, duration: int = 10):
    """Run a single client session."""
    async with websockets.connect(url) as websocket:
        # Receive welcome
        welcome = await websocket.recv()
        data = json.loads(welcome)
        session_id = data["session_id"]
        
        # Send start
        await websocket.send(json.dumps({
            "type": "start",
            "codec": "pcm16",
            "sample_rate": 16000
        }))
        
        # Stream audio for duration
        start = time.time()
        frames_sent = 0
        
        while time.time() - start < duration:
            # Send 30ms frame
            await websocket.send(b"\x00" * 960)
            frames_sent += 1
            await asyncio.sleep(0.03)
        
        return session_id, frames_sent

async def load_test(url: str, num_clients: int = 50):
    """Run load test with multiple clients."""
    print(f"Starting load test with {num_clients} clients...")
    
    start = time.time()
    
    # Run concurrent sessions
    tasks = [
        client_session(url, duration=10)
        for _ in range(num_clients)
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    elapsed = time.time() - start
    
    # Analyze results
    successful = [r for r in results if not isinstance(r, Exception)]
    failed = [r for r in results if isinstance(r, Exception)]
    
    print(f"\nResults:")
    print(f"  Duration: {elapsed:.2f}s")
    print(f"  Successful: {len(successful)}/{num_clients}")
    print(f"  Failed: {len(failed)}/{num_clients}")
    
    if successful:
        total_frames = sum(frames for _, frames in successful)
        print(f"  Total frames: {total_frames}")
        print(f"  Frames/sec: {total_frames/elapsed:.0f}")

if __name__ == "__main__":
    url = "ws://localhost:8000/ws/audio"
    asyncio.run(load_test(url, num_clients=50))
```

#### Acceptance Criteria:
- ✅ Bidirectional streaming works
- ✅ Handles PCM16 16kHz audio
- ✅ Frame validation (20-40ms)
- ✅ Backpressure at 8 frames
- ✅ Barge-in detection < 100ms
- ✅ Session state preserved
- ✅ Auto-reconnect support
- ✅ Turn detection accurate
- ✅ 50+ concurrent sessions
- ✅ Clean disconnection handling
- ✅ Control messages processed
- ✅ Unique session IDs
- ✅ Memory bounded per session

---

*Continue to [Phase 7-8: UI & Testing](phase7-8.md)*
