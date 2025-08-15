from pydantic import BaseModel

class TranscriptChunk(BaseModel):
    text: str
    is_final: bool
    start_ms: int
    end_ms: int
    confidence: float

class TokenChunk(BaseModel):
    token: str
    logprob: float | None
    is_final: bool

class AudioChunk(BaseModel):
    format: str
    sample_rate: int
    payload: bytes
