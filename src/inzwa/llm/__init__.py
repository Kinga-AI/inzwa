"""LLM module for text generation."""

from typing import AsyncIterator, List, Dict, Any

from ..models import TokenChunk
from .engine import LLMEngine
from .safety import SafetyFilter


async def generate_stream(messages: List[Dict[str, str]]) -> AsyncIterator[TokenChunk]:
    """Stream generation of tokens."""
    engine = LLMEngine()
    async for chunk in engine.generate_stream(messages):
        yield chunk


async def generate_batch(messages: List[Dict[str, str]]) -> str:
    """Batch generation of complete response."""
    engine = LLMEngine()
    return await engine.generate_batch(messages)


__all__ = [
    "generate_stream",
    "generate_batch",
    "LLMEngine",
    "SafetyFilter",
    "TokenChunk"
]
