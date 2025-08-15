"""Session orchestrator for managing the real-time pipeline."""

import asyncio
import uuid
from typing import Optional
from fastapi import WebSocket

from ..asr import ASREngine
from ..llm import LLMEngine
from ..tts import TTSEngine
from ..utils.audio import AudioBuffer
from ..telemetry import get_logger
from .state import DialogueState
from .backpressure import BackpressureManager

logger = get_logger(__name__)


class SessionOrchestrator:
    """Orchestrates the ASR -> LLM -> TTS pipeline for a session."""
    
    def __init__(
        self,
        session_id: str,
        websocket: WebSocket,
        codec: str = "pcm16",
        sample_rate: int = 16000
    ):
        self.session_id = session_id
        self.websocket = websocket
        self.codec = codec
        self.sample_rate = sample_rate
        
        # Initialize components
        self.asr_engine = ASREngine()
        self.llm_engine = LLMEngine()
        self.tts_engine = TTSEngine()
        
        # Session state
        self.dialogue_state = DialogueState()
        self.audio_buffer = AudioBuffer(sample_rate)
        self.backpressure = BackpressureManager()
        
        # Control flags
        self.is_running = False
        self.is_processing = False
        
        # Async queues for streaming
        self.audio_queue = asyncio.Queue(maxsize=100)
        self.transcript_queue = asyncio.Queue(maxsize=50)
        self.token_queue = asyncio.Queue(maxsize=100)
        self.audio_out_queue = asyncio.Queue(maxsize=100)
        
        # Tasks
        self.tasks = []
    
    async def run(self):
        """Run the orchestration pipeline."""
        self.is_running = True
        logger.info(f"Starting orchestrator for session {self.session_id}")
        
        try:
            # Start pipeline tasks
            self.tasks = [
                asyncio.create_task(self._asr_task()),
                asyncio.create_task(self._llm_task()),
                asyncio.create_task(self._tts_task()),
                asyncio.create_task(self._output_task())
            ]
            
            # Wait for all tasks
            await asyncio.gather(*self.tasks)
            
        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
        finally:
            await self.cleanup()
    
    async def process_audio(self, audio_bytes: bytes):
        """Process incoming audio bytes."""
        if not self.is_running:
            return
        
        # Apply backpressure if needed
        if self.backpressure.should_drop(self.audio_queue.qsize()):
            logger.warning(f"Dropping audio frame due to backpressure")
            return
        
        await self.audio_queue.put(audio_bytes)
    
    async def _asr_task(self):
        """ASR processing task."""
        try:
            while self.is_running:
                # Get audio from queue
                audio_bytes = await self.audio_queue.get()
                
                # Add to buffer
                self.audio_buffer.add(audio_bytes)
                
                # Process with VAD
                if self.audio_buffer.has_speech():
                    # Stream transcription
                    async for transcript in self.asr_engine.transcribe_stream(
                        self.audio_buffer.get_chunk()
                    ):
                        await self.transcript_queue.put(transcript)
                        
                        # Send partial transcript to client
                        await self.websocket.send_json({
                            "type": "asr.partial",
                            "text": transcript.text,
                            "start_ms": transcript.start_ms,
                            "end_ms": transcript.end_ms,
                            "conf": transcript.confidence
                        })
        except Exception as e:
            logger.error(f"ASR task error: {e}")
    
    async def _llm_task(self):
        """LLM processing task."""
        try:
            while self.is_running:
                # Collect transcripts until end of utterance
                transcript = await self.transcript_queue.get()
                
                if transcript.is_final:
                    # Update dialogue state
                    self.dialogue_state.add_user_turn(transcript.text)
                    
                    # Generate response
                    messages = self.dialogue_state.get_messages()
                    
                    response_text = ""
                    async for token in self.llm_engine.generate_stream(messages):
                        await self.token_queue.put(token)
                        response_text += token.token
                        
                        # Send token to client
                        await self.websocket.send_json({
                            "type": "llm.partial",
                            "token": token.token
                        })
                    
                    # Update dialogue state with assistant response
                    self.dialogue_state.add_assistant_turn(response_text)
        
        except Exception as e:
            logger.error(f"LLM task error: {e}")
    
    async def _tts_task(self):
        """TTS processing task."""
        try:
            phrase_buffer = ""
            
            while self.is_running:
                # Collect tokens into phrases
                token = await self.token_queue.get()
                phrase_buffer += token.token
                
                # Check for phrase boundary
                if self._is_phrase_boundary(phrase_buffer):
                    # Synthesize phrase
                    await self.websocket.send_json({"type": "tts.start"})
                    
                    async for audio_chunk in self.tts_engine.synthesize_stream(phrase_buffer):
                        await self.audio_out_queue.put(audio_chunk)
                    
                    phrase_buffer = ""
                    await self.websocket.send_json({"type": "tts.end"})
        
        except Exception as e:
            logger.error(f"TTS task error: {e}")
    
    async def _output_task(self):
        """Output audio streaming task."""
        try:
            while self.is_running:
                audio_chunk = await self.audio_out_queue.get()
                
                # Send audio to client
                if self.codec == "opus":
                    await self.websocket.send_bytes(audio_chunk.payload)
                else:
                    # Convert if needed
                    await self.websocket.send_bytes(audio_chunk.payload)
        
        except Exception as e:
            logger.error(f"Output task error: {e}")
    
    def _is_phrase_boundary(self, text: str) -> bool:
        """Check if text ends at a phrase boundary."""
        # Simple heuristic: punctuation or length
        return any(text.endswith(p) for p in ['.', '!', '?', ',']) or len(text) > 50
    
    async def end_turn(self):
        """End the current turn."""
        # Flush buffers and process remaining audio
        if self.audio_buffer.has_data():
            # Process remaining audio
            pass
    
    async def stop(self):
        """Stop the orchestrator."""
        self.is_running = False
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        await self.cleanup()
    
    async def cleanup(self):
        """Clean up resources."""
        logger.info(f"Cleaning up session {self.session_id}")
        # TODO: Clean up model resources if needed
