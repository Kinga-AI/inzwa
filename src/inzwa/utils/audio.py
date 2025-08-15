"""Audio processing utilities."""

import numpy as np
from typing import Optional, Tuple
import io


class AudioBuffer:
    """Buffer for streaming audio data."""
    
    def __init__(self, sample_rate: int = 16000, chunk_duration_ms: int = 20):
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        self.buffer = bytearray()
        self.speech_buffer = bytearray()
    
    def add(self, audio_bytes: bytes):
        """Add audio bytes to buffer."""
        self.buffer.extend(audio_bytes)
    
    def has_data(self) -> bool:
        """Check if buffer has data."""
        return len(self.buffer) > 0
    
    def has_speech(self) -> bool:
        """Check if buffer contains speech (simple energy-based)."""
        if len(self.buffer) < self.chunk_size * 2:  # Need at least 2 chunks
            return False
        
        # Convert to numpy for energy calculation
        audio = np.frombuffer(self.buffer, dtype=np.int16)
        energy = np.mean(np.abs(audio))
        
        # Simple threshold (can be improved with proper VAD)
        return energy > 500
    
    def get_chunk(self) -> bytes:
        """Get a chunk of audio data."""
        if len(self.buffer) >= self.chunk_size * 2:
            chunk = bytes(self.buffer[:self.chunk_size * 2])
            self.buffer = self.buffer[self.chunk_size * 2:]
            return chunk
        return bytes(self.buffer)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
        self.speech_buffer.clear()


def convert_audio_format(
    audio_data: bytes,
    from_format: str,
    to_format: str,
    sample_rate: int = 16000
) -> bytes:
    """Convert audio between formats."""
    
    if from_format == to_format:
        return audio_data
    
    # Placeholder conversions
    if from_format == "pcm16" and to_format == "opus":
        # TODO: Implement PCM to Opus conversion
        return audio_data
    
    if from_format == "opus" and to_format == "pcm16":
        # TODO: Implement Opus to PCM conversion
        return audio_data
    
    return audio_data


def resample_audio(
    audio_data: np.ndarray,
    orig_sr: int,
    target_sr: int
) -> np.ndarray:
    """Resample audio to target sample rate."""
    
    if orig_sr == target_sr:
        return audio_data
    
    try:
        import librosa
        return librosa.resample(
            audio_data.astype(np.float32),
            orig_sr=orig_sr,
            target_sr=target_sr
        )
    except ImportError:
        # Simple linear interpolation fallback
        ratio = target_sr / orig_sr
        new_length = int(len(audio_data) * ratio)
        indices = np.linspace(0, len(audio_data) - 1, new_length)
        return np.interp(indices, np.arange(len(audio_data)), audio_data)
