"""WebSocket manager for real-time audio streaming."""

import asyncio
import json
import uuid
from typing import Dict, Set
from fastapi import WebSocket
from ..orchestration import SessionOrchestrator
from ..telemetry import get_logger

logger = get_logger(__name__)


class WebSocketManager:
    """Manages WebSocket connections and audio streaming sessions."""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.sessions: Dict[str, SessionOrchestrator] = {}
    
    async def connect(self, websocket: WebSocket):
        """Accept WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        self.active_connections.discard(websocket)
        # Clean up associated session
        for session_id, orchestrator in list(self.sessions.items()):
            if orchestrator.websocket == websocket:
                del self.sessions[session_id]
                logger.info(f"Cleaned up session {session_id}")
                break
    
    async def handle_session(self, websocket: WebSocket):
        """Handle a WebSocket audio streaming session."""
        session_id = None
        orchestrator = None
        
        try:
            while True:
                # Receive message (can be binary audio or JSON control)
                message = await websocket.receive()
                
                if "text" in message:
                    # JSON control message
                    data = json.loads(message["text"])
                    msg_type = data.get("type")
                    
                    if msg_type == "start":
                        # Start new session
                        session_id = data.get("session_id", str(uuid.uuid4()))
                        codec = data.get("codec", "pcm16")
                        sample_rate = data.get("sample_rate", 16000)
                        
                        orchestrator = SessionOrchestrator(
                            session_id=session_id,
                            websocket=websocket,
                            codec=codec,
                            sample_rate=sample_rate
                        )
                        self.sessions[session_id] = orchestrator
                        
                        # Start orchestrator processing
                        asyncio.create_task(orchestrator.run())
                        
                        # Send acknowledgment
                        await websocket.send_json({
                            "type": "session.started",
                            "session_id": session_id
                        })
                        logger.info(f"Started session {session_id}")
                    
                    elif msg_type == "end_turn":
                        # End current turn
                        if orchestrator:
                            await orchestrator.end_turn()
                    
                    elif msg_type == "stop":
                        # Stop session
                        if orchestrator:
                            await orchestrator.stop()
                        break
                
                elif "bytes" in message:
                    # Binary audio data
                    if orchestrator:
                        await orchestrator.process_audio(message["bytes"])
                    else:
                        logger.warning("Received audio without active session")
        
        except Exception as e:
            logger.error(f"WebSocket session error: {e}")
            if orchestrator:
                await orchestrator.stop()
        
        finally:
            if session_id and session_id in self.sessions:
                del self.sessions[session_id]
