"""Ultra-light settings per .cursorrules."""

from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Minimal settings for Inzwa - no overengineering."""
    
    model_config = SettingsConfigDict(env_prefix="INZWA_", case_sensitive=False)
    
    # Core settings
    debug: bool = False
    cors_allowed_origins: str = "http://localhost:7860"
    request_timeout_s: float = 5.0
    max_text_chars: int = 400
    max_audio_seconds: int = 20
    
    # Feature flags
    enable_webrtc: bool = False
    
    # ASR settings
    asr_engine: str = "faster-whisper"   # or "whisper.cpp"
    asr_model: str = "small"
    
    # LLM settings
    llm_engine: str = "llama-cpp"        # or "vllm"
    llm_model: str = "mistral-2b-shona-lora"
    
    # TTS settings
    tts_engine: str = "coqui-vits-lite"
    audio_out_codec: str = "opus"
    
    # Performance
    max_concurrent_sessions: int = 50
    backpressure_threshold: int = 8  # Per .cursorrules: max_queue=8
    
    # Security
    require_auth: bool = False
    api_key: str | None = None


# Single instance - injected via DI, not global
settings = Settings()
