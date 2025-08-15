"""TTS API router."""

from typing import Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ...tts import synthesize_stream, synthesize_batch
from ...telemetry import tts_rtf_histogram

router = APIRouter()


class TTSRequest(BaseModel):
    text: str
    voice: str = "shona_female_a"
    format: str = "opus"  # or "wav", "pcm16"
    stream: bool = True


@router.post("/tts")
async def text_to_speech(request: TTSRequest):
    """Convert text to speech."""
    try:
        if request.stream:
            # Streaming audio response
            async def generate():
                with tts_rtf_histogram.time():
                    async for chunk in synthesize_stream(
                        request.text,
                        voice=request.voice,
                        format=request.format
                    ):
                        yield chunk.payload
            
            media_type = "audio/opus" if request.format == "opus" else "audio/wav"
            return StreamingResponse(
                generate(),
                media_type=media_type
            )
        else:
            # Non-streaming response
            audio_data = await synthesize_batch(
                request.text,
                voice=request.voice,
                format=request.format
            )
            return StreamingResponse(
                iter([audio_data]),
                media_type="audio/wav"
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
