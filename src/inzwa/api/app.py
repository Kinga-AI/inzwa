"""Inzwa Gateway API - Main FastAPI application."""

from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from prometheus_client import make_asgi_app

from ..config import settings
from ..telemetry import setup_logging
from .routers import asr_router, chat_router, tts_router, admin_router
from .websocket import WebSocketManager
from .auth import verify_api_key

# Setup logging
logger = setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app lifecycle - model loading, cleanup, etc."""
    logger.info("Starting Inzwa API...")
    # TODO: Load models on startup if warmup enabled
    yield
    logger.info("Shutting down Inzwa API...")
    # TODO: Cleanup resources

app = FastAPI(
    title="Inzwa API",
    version="0.1.0",
    description="Real-time Shona Speech-to-Speech Assistant",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket manager
ws_manager = WebSocketManager()

# Mount Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Include routers
app.include_router(asr_router, prefix="/v1", tags=["ASR"])
app.include_router(chat_router, prefix="/v1", tags=["Chat"])
app.include_router(tts_router, prefix="/v1", tags=["TTS"])
app.include_router(admin_router, prefix="/v1/admin", tags=["Admin"])

@app.get("/healthz")
async def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}

@app.get("/v1/models")
async def list_models() -> Dict[str, Any]:
    """List available models and their versions."""
    return {
        "asr": [
            {"name": settings.asr_model, "engine": settings.asr_engine, "loaded": False}
        ],
        "llm": [
            {"name": settings.llm_model, "engine": settings.llm_engine, "loaded": False}
        ],
        "tts": [
            {"name": settings.tts_model, "engine": settings.tts_engine, "loaded": False}
        ]
    }

@app.websocket("/ws/audio")
async def websocket_audio_endpoint(
    websocket: WebSocket,
    token: str = Depends(verify_api_key)
):
    """WebSocket endpoint for real-time audio streaming."""
    await ws_manager.connect(websocket)
    try:
        await ws_manager.handle_session(websocket)
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
        logger.info("WebSocket disconnected")

# WebRTC endpoint placeholder
@app.post("/rtc/session")
async def webrtc_session():
    """WebRTC session establishment."""
    # TODO: Implement WebRTC signaling with aiortc
    raise HTTPException(status_code=501, detail="WebRTC not yet implemented")
