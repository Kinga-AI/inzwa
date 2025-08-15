"""Authentication and authorization utilities."""

from typing import Optional
from fastapi import HTTPException, Header
from ..config import settings


async def verify_api_key(authorization: Optional[str] = Header(None)) -> str:
    """Verify API key from Authorization header."""
    if not settings.require_auth:
        return "anonymous"
    
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    # Extract token from "Bearer <token>"
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid authorization format")
    
    token = parts[1]
    
    # TODO: Implement actual token verification
    # For now, just check against a simple env var
    if token != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return token
