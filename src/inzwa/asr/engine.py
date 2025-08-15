"""ASR engine implementation with pluggable backends."""

import asyncio
from typing import AsyncIterator, Dict, Any, Optional
from ..config import settings
from ..models import TranscriptChunk
from ..telemetry import get_logger

logger = get_logger(__name__)


class ASREngine:
    """ASR engine with support for faster-whisper and whisper.cpp."""
    
    def __init__(self):
        self.engine_type = settings.asr_engine
        self.model_name = settings.asr_model
        self.device = settings.asr_device
        self.compute_type = settings.asr_compute_type
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the ASR model based on configuration."""
        if self.engine_type == "faster-whisper":
            self._load_faster_whisper()
        elif self.engine_type == "whisper.cpp":
            self._load_whisper_cpp()
        else:
            raise ValueError(f"Unknown ASR engine: {self.engine_type}")
    
    def _load_faster_whisper(self):
        """Load faster-whisper model."""
        try:
            from faster_whisper import WhisperModel
            
            model_path = settings.asr_model_path or self.model_name
            self.model = WhisperModel(
                model_path,
                device=self.device,
                compute_type=self.compute_type
            )
            logger.info(f"Loaded faster-whisper model: {model_path}")
        except ImportError:
            logger.error("faster-whisper not installed")
            # Fallback to placeholder
            self.model = None
    
    def _load_whisper_cpp(self):
        """Load whisper.cpp model."""
        # TODO: Implement whisper.cpp loading
        logger.warning("whisper.cpp backend not yet implemented")
        self.model = None
    
    async def transcribe_stream(
        self,
        audio_frames: bytes,
        language: str = "sn"  # Shona
    ) -> AsyncIterator[TranscriptChunk]:
        """Stream transcription with partial results."""
        
        if self.model is None:
            # Placeholder implementation
            yield TranscriptChunk(
                text="[ASR not loaded]",
                is_final=True,
                start_ms=0,
                end_ms=1000,
                confidence=0.0
            )
            return
        
        # Run transcription in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        if self.engine_type == "faster-whisper":
            # Streaming with faster-whisper
            segments, info = await loop.run_in_executor(
                None,
                self._transcribe_faster_whisper,
                audio_frames,
                language
            )
            
            for segment in segments:
                yield TranscriptChunk(
                    text=segment.text.strip(),
                    is_final=False,
                    start_ms=int(segment.start * 1000),
                    end_ms=int(segment.end * 1000),
                    confidence=segment.avg_logprob if hasattr(segment, 'avg_logprob') else 0.9
                )
    
    def _transcribe_faster_whisper(self, audio: bytes, language: str):
        """Run faster-whisper transcription."""
        if not self.model:
            return [], None
        
        # Convert bytes to numpy array
        import numpy as np
        audio_array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        
        segments, info = self.model.transcribe(
            audio_array,
            language=language,
            beam_size=5,
            vad_filter=settings.asr_vad_enabled
        )
        
        return list(segments), info
    
    async def transcribe_batch(
        self,
        audio_bytes: bytes,
        sample_rate: int = 16000
    ) -> Dict[str, Any]:
        """Batch transcription of complete audio."""
        
        full_text = ""
        segments = []
        
        async for chunk in self.transcribe_stream(audio_bytes):
            full_text += chunk.text + " "
            segments.append({
                "text": chunk.text,
                "start_ms": chunk.start_ms,
                "end_ms": chunk.end_ms,
                "conf": chunk.confidence
            })
        
        return {
            "text": full_text.strip(),
            "segments": segments
        }
