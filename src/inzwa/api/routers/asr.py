"""ASR API router."""

import base64
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...asr import transcribe_batch
from ...telemetry import asr_latency

router = APIRouter()


class ASRRequest(BaseModel):
    audio_base64: str
    sample_rate: int = 16000


class ASRResponse(BaseModel):
    text: str
    segments: list[Dict[str, Any]]


@router.post("/asr", response_model=ASRResponse)
async def transcribe_audio(request: ASRRequest) -> ASRResponse:
    """Transcribe audio to text."""
    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(request.audio_base64)
        
        # Track latency
        with asr_latency.time():
            result = await transcribe_batch(audio_bytes, request.sample_rate)
        
        return ASRResponse(
            text=result["text"],
            segments=result["segments"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
