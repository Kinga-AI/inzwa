"""API routers for different endpoints."""

from .asr import router as asr_router
from .chat import router as chat_router
from .tts import router as tts_router
from .admin import router as admin_router

__all__ = ["asr_router", "chat_router", "tts_router", "admin_router"]
