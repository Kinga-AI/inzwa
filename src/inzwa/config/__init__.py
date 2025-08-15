"""Configuration management using pydantic-settings."""

from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: List[str] = ["*"]
    require_auth: bool = False
    api_key: Optional[str] = None
    
    # WebRTC/WebSocket
    enable_webrtc: bool = True
    enable_websocket: bool = True
    audio_chunk_ms: int = 20  # 20-40ms chunks
    
    # ASR Settings
    asr_engine: str = "faster-whisper"  # or "whisper.cpp"
    asr_model: str = "small"  # small/base for low latency
    asr_model_path: Optional[str] = None
    asr_device: str = "cpu"  # cpu/cuda
    asr_compute_type: str = "int8"  # int8/float16/float32
    asr_vad_enabled: bool = True
    asr_vad_model: str = "silero"  # silero/webrtc
    
    # LLM Settings
    llm_engine: str = "llama-cpp"  # llama-cpp/vllm
    llm_model: str = "mistral-2b-shona-lora"
    llm_model_path: Optional[str] = None
    llm_device: str = "cpu"
    llm_max_tokens: int = 512
    llm_temperature: float = 0.7
    llm_context_size: int = 4096
    llm_quantization: str = "Q4_K_M"  # for llama-cpp
    llm_gpu_layers: int = 0  # for llama-cpp GPU offload
    
    # TTS Settings
    tts_engine: str = "coqui-vits-lite"
    tts_model: str = "shona_female_a"
    tts_model_path: Optional[str] = None
    tts_device: str = "cpu"
    tts_use_onnx: bool = True  # Use ONNX for faster inference
    audio_out_codec: str = "opus"  # opus/pcm16
    
    # Session Management
    max_concurrent_sessions: int = 50
    session_timeout_seconds: int = 60
    
    # Performance
    streaming_enabled: bool = True
    backpressure_threshold: int = 100  # Queue size threshold
    gpu_policy: str = "prefer"  # prefer/require/disable
    quantization_level: str = "aggressive"  # aggressive/moderate/none
    
    # Observability
    log_level: str = "INFO"
    metrics_enabled: bool = True
    tracing_enabled: bool = False
    
    # Model Registry
    model_cache_dir: str = "./models"
    model_warmup_on_start: bool = False
    huggingface_hub_token: Optional[str] = None
    
    # Data Collection (opt-in)
    data_collection_enabled: bool = False
    data_retention_days: int = 7
    
    # Feature Flags
    enable_rag: bool = False
    enable_safety_filters: bool = True
    enable_speculative_decoding: bool = False
    enable_edge_inference: bool = False


# Global settings instance
settings = Settings()


# Export settings
__all__ = ["settings", "Settings"]
