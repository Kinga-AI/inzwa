"""TTS module for speech synthesis."""

from typing import AsyncIterator, Optional

from ..models import AudioChunk
from .engine import TTSEngine


async def synthesize_stream(
    text: str,
    voice: str = "shona_female_a",
    format: str = "opus"
) -> AsyncIterator[AudioChunk]:
    """Stream synthesis of audio."""
    engine = TTSEngine()
    async for chunk in engine.synthesize_stream(text, voice, format):
        yield chunk


async def synthesize_batch(
    text: str,
    voice: str = "shona_female_a",
    format: str = "wav"
) -> bytes:
    """Batch synthesis of complete audio."""
    engine = TTSEngine()
    return await engine.synthesize_batch(text, voice, format)


__all__ = [
    "synthesize_stream",
    "synthesize_batch",
    "TTSEngine",
    "AudioChunk"
]
