"""Admin API router for model management."""

from fastapi import APIRouter, Depends, HTTPException
from ...config import settings
from ...models.registry import ModelRegistry
from ..auth import verify_api_key

router = APIRouter()

# Model registry instance
model_registry = ModelRegistry()


@router.post("/warmup", dependencies=[Depends(verify_api_key)])
async def warmup_models():
    """Pre-load and warm up all models."""
    try:
        results = await model_registry.warmup_all()
        return {
            "status": "success",
            "models": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Warmup failed: {str(e)}")


@router.post("/reload/{model_type}", dependencies=[Depends(verify_api_key)])
async def reload_model(model_type: str):
    """Reload a specific model."""
    if model_type not in ["asr", "llm", "tts"]:
        raise HTTPException(status_code=400, detail="Invalid model type")
    
    try:
        result = await model_registry.reload_model(model_type)
        return {"status": "success", "model": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {str(e)}")


@router.get("/status")
async def get_status():
    """Get system status and loaded models."""
    return {
        "status": "running",
        "models": model_registry.get_loaded_models(),
        "sessions": {
            "active": 0,  # TODO: Get from session manager
            "max": settings.max_concurrent_sessions
        }
    }
