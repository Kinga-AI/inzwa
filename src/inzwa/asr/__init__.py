"""ASR module for speech recognition."""

from typing import AsyncIterator, Dict, Any

from ..models import TranscriptChunk
from .engine import ASREngine
from .vad import VADProcessor


async def transcribe_stream(frames: bytes) -> AsyncIterator[TranscriptChunk]:
    """Stream transcription of audio frames."""
    engine = ASREngine()
    async for chunk in engine.transcribe_stream(frames):
        yield chunk


async def transcribe_batch(audio_bytes: bytes, sample_rate: int = 16000) -> Dict[str, Any]:
    """Batch transcription of complete audio."""
    engine = ASREngine()
    return await engine.transcribe_batch(audio_bytes, sample_rate)


__all__ = [
    "transcribe_stream",
    "transcribe_batch",
    "ASREngine",
    "VADProcessor",
    "TranscriptChunk"
]
