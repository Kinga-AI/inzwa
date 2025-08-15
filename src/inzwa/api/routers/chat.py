"""Chat/LLM API router."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json

from ...llm import generate_stream, generate_batch
from ...telemetry import llm_ttfw_histogram

router = APIRouter()


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    stream: bool = True
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.7


@router.post("/chat")
async def chat_completion(request: ChatRequest):
    """Generate chat completion."""
    try:
        messages = [msg.dict() for msg in request.messages]
        
        if request.stream:
            # Streaming response
            async def generate():
                with llm_ttfw_histogram.time():
                    first_token = True
                    async for chunk in generate_stream(messages):
                        if first_token:
                            first_token = False
                        yield f"data: {json.dumps({'token': chunk.token})}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                generate(),
                media_type="text/event-stream"
            )
        else:
            # Non-streaming response
            result = await generate_batch(messages)
            return {"content": result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
