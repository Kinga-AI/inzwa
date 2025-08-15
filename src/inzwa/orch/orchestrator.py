"""Minimal orchestrator per .cursorrules template - <50 lines."""

from __future__ import annotations
import asyncio
from typing import AsyncIterator
from ..models import TranscriptChunk, AudioChunk, TokenChunk


class Orchestrator:
    """Ultra-light orchestrator - no overengineering."""
    
    def __init__(self, asr, llm, tts, *, max_queue: int = 8):
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.max_queue = max_queue
    
    async def run(self, frames: AsyncIterator[bytes]) -> AsyncIterator[AudioChunk]:
        """Minimal pipeline: ASR → LLM → TTS."""
        # Process audio frames through ASR
        async for partial in self.asr.transcribe_stream(frames):
            if not partial.text:
                continue
            
            # Check for phrase boundary
            if partial.is_final or self._is_phrase_end(partial.text):
                # Generate LLM response
                response = await self._generate_response(partial.text)
                
                # Synthesize audio
                async for audio in self.tts.synthesize_stream(response):
                    yield audio
    
    def _is_phrase_end(self, text: str) -> bool:
        """Simple phrase boundary detection."""
        return any(text.endswith(p) for p in (".", "?", "!", ","))
    
    async def _generate_response(self, text: str) -> AsyncIterator[str]:
        """Generate LLM response with phrase chunking."""
        buffer = ""
        messages = [{"role": "user", "content": text}]
        
        async for tok in self.llm.generate_stream(messages):
            buffer += tok.token
            if tok.is_final or self._is_phrase_end(buffer):
                yield buffer
                buffer = ""
        
        if buffer:
            yield buffer
