"""Voice Activity Detection for ASR."""

import numpy as np
from typing import Tuple, Optional
from ..config import settings
from ..telemetry import get_logger

logger = get_logger(__name__)


class VADProcessor:
    """Voice Activity Detection processor."""
    
    def __init__(self, model_type: str = None):
        self.model_type = model_type or settings.asr_vad_model
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load VAD model based on type."""
        if self.model_type == "silero":
            self._load_silero_vad()
        elif self.model_type == "webrtc":
            self._load_webrtc_vad()
        else:
            logger.warning(f"Unknown VAD model: {self.model_type}")
    
    def _load_silero_vad(self):
        """Load Silero VAD model."""
        try:
            import torch
            # Load Silero VAD
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False
            )
            self.model = model
            self.get_speech_timestamps = utils[0]
            logger.info("Loaded Silero VAD")
        except Exception as e:
            logger.error(f"Failed to load Silero VAD: {e}")
            self.model = None
    
    def _load_webrtc_vad(self):
        """Load WebRTC VAD."""
        try:
            import webrtcvad
            self.model = webrtcvad.Vad(2)  # Aggressiveness level 2
            logger.info("Loaded WebRTC VAD")
        except ImportError:
            logger.error("webrtcvad not installed")
            self.model = None
    
    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """Process audio and detect speech."""
        
        if self.model is None:
            # No VAD, assume all audio contains speech
            return True, audio
        
        if self.model_type == "silero":
            return self._process_silero(audio, sample_rate)
        elif self.model_type == "webrtc":
            return self._process_webrtc(audio, sample_rate)
        
        return True, audio
    
    def _process_silero(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """Process with Silero VAD."""
        if not self.model:
            return True, audio
        
        # Get speech timestamps
        speech_timestamps = self.get_speech_timestamps(
            audio,
            self.model,
            sampling_rate=sample_rate
        )
        
        if not speech_timestamps:
            return False, None
        
        # Extract speech segments
        # For simplicity, return the whole audio if speech detected
        return True, audio
    
    def _process_webrtc(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """Process with WebRTC VAD."""
        if not self.model:
            return True, audio
        
        # Convert to bytes
        audio_bytes = (audio * 32768).astype(np.int16).tobytes()
        
        # Process in frames (30ms)
        frame_duration_ms = 30
        frame_length = int(sample_rate * frame_duration_ms / 1000)
        
        has_speech = False
        for i in range(0, len(audio_bytes) - frame_length, frame_length):
            frame = audio_bytes[i:i + frame_length]
            if self.model.is_speech(frame, sample_rate):
                has_speech = True
                break
        
        return has_speech, audio if has_speech else None
