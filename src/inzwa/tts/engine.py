"""TTS engine implementation with Coqui TTS."""

import asyncio
import io
from typing import AsyncIterator, Optional
import numpy as np
from ..config import settings
from ..models import AudioChunk
from ..telemetry import get_logger, tts_rtf_histogram

logger = get_logger(__name__)


class TTSEngine:
    """TTS engine using Coqui TTS."""
    
    def __init__(self):
        self.engine_type = settings.tts_engine
        self.model_name = settings.tts_model
        self.device = settings.tts_device
        self.use_onnx = settings.tts_use_onnx
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the TTS model."""
        if "coqui" in self.engine_type.lower():
            self._load_coqui_tts()
        else:
            raise ValueError(f"Unknown TTS engine: {self.engine_type}")
    
    def _load_coqui_tts(self):
        """Load Coqui TTS model."""
        try:
            from TTS.api import TTS
            
            model_path = settings.tts_model_path
            
            if model_path:
                # Load custom model
                self.model = TTS(model_path=model_path, gpu=(self.device == "cuda"))
            else:
                # Use default or download model
                # For Shona, we'd use a custom trained model
                logger.warning("Using placeholder TTS - custom Shona model needed")
                self.model = None
            
            if self.model:
                logger.info(f"Loaded Coqui TTS model")
        except ImportError:
            logger.error("Coqui TTS not installed")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to load Coqui TTS: {e}")
            self.model = None
    
    async def synthesize_stream(\n        self,\n        text: str,\n        voice: str = \"shona_female_a\",\n        format: str = \"opus\"\n    ) -> AsyncIterator[AudioChunk]:\n        """Stream synthesis of audio chunks.\"\"\"\n        \n        if self.model is None:\n            # Placeholder implementation\n            yield AudioChunk(\n                format=format,\n                sample_rate=48000 if format == \"opus\" else 16000,\n                payload=b\"\"  # Empty audio\n            )\n            return\n        \n        # Run synthesis in thread pool\n        loop = asyncio.get_event_loop()\n        \n        with tts_rtf_histogram.time():\n            audio_data = await loop.run_in_executor(\n                None,\n                self._synthesize_coqui,\n                text,\n                voice\n            )\n        \n        if audio_data is None:\n            return\n        \n        # Convert and stream audio chunks\n        chunk_size = 1024  # Adjust based on latency requirements\n        \n        for i in range(0, len(audio_data), chunk_size):\n            chunk = audio_data[i:i + chunk_size]\n            \n            # Convert format if needed\n            if format == \"opus\":\n                chunk = self._convert_to_opus(chunk)\n            \n            yield AudioChunk(\n                format=format,\n                sample_rate=48000 if format == \"opus\" else 16000,\n                payload=chunk\n            )\n    \n    def _synthesize_coqui(self, text: str, voice: str) -> Optional[bytes]:\n        \"\"\"Synthesize with Coqui TTS.\"\"\"\n        if not self.model:\n            return None\n        \n        try:\n            # Generate audio\n            wav = self.model.tts(\n                text=text,\n                speaker=voice if voice != \"shona_female_a\" else None,\n                language=\"en\"  # Placeholder - should be \"sn\" for Shona\n            )\n            \n            # Convert to bytes\n            if isinstance(wav, np.ndarray):\n                # Convert float32 [-1, 1] to int16\n                audio_int16 = (wav * 32767).astype(np.int16)\n                return audio_int16.tobytes()\n            \n            return wav\n        \n        except Exception as e:\n            logger.error(f\"TTS synthesis error: {e}\")\n            return None\n    \n    def _convert_to_opus(self, audio_bytes: bytes) -> bytes:\n        \"\"\"Convert audio to Opus format.\"\"\"\n        # TODO: Implement Opus encoding\n        # For now, return PCM\n        return audio_bytes\n    \n    async def synthesize_batch(\n        self,\n        text: str,\n        voice: str = \"shona_female_a\",\n        format: str = \"wav\"\n    ) -> bytes:\n        \"\"\"Batch synthesis of complete audio.\"\"\"\n        \n        full_audio = b\"\"\n        \n        async for chunk in self.synthesize_stream(text, voice, format):\n            full_audio += chunk.payload\n        \n        if format == \"wav\":\n            # Add WAV header\n            full_audio = self._add_wav_header(full_audio, 16000)\n        \n        return full_audio\n    \n    def _add_wav_header(self, audio_bytes: bytes, sample_rate: int) -> bytes:\n        \"\"\"Add WAV header to raw audio bytes.\"\"\"\n        import wave\n        \n        buffer = io.BytesIO()\n        with wave.open(buffer, 'wb') as wav:\n            wav.setnchannels(1)  # Mono\n            wav.setsampwidth(2)  # 16-bit\n            wav.setframerate(sample_rate)\n            wav.writeframes(audio_bytes)\n        \n        return buffer.getvalue()
