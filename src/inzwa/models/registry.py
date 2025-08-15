"""Model registry for managing ML models."""

import os
from typing import Dict, Any, Optional
from ..config import settings
from ..telemetry import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """Registry for managing and loading models."""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.model_paths: Dict[str, str] = {}
        self._setup_paths()
    
    def _setup_paths(self):
        """Setup model paths from config."""
        self.model_paths = {
            "asr": settings.asr_model_path or f"{settings.model_cache_dir}/asr/{settings.asr_model}",
            "llm": settings.llm_model_path or f"{settings.model_cache_dir}/llm/{settings.llm_model}",
            "tts": settings.tts_model_path or f"{settings.model_cache_dir}/tts/{settings.tts_model}"
        }
        
        # Ensure directories exist
        for path in self.model_paths.values():
            os.makedirs(os.path.dirname(path), exist_ok=True)
    
    async def warmup_all(self) -> Dict[str, Any]:
        """Warm up all models."""
        results = {}
        
        # Load ASR
        try:
            from ..asr import ASREngine
            self.models["asr"] = ASREngine()
            results["asr"] = {"status": "loaded", "model": settings.asr_model}
        except Exception as e:
            logger.error(f"Failed to load ASR: {e}")
            results["asr"] = {"status": "failed", "error": str(e)}
        
        # Load LLM
        try:
            from ..llm import LLMEngine
            self.models["llm"] = LLMEngine()
            results["llm"] = {"status": "loaded", "model": settings.llm_model}
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            results["llm"] = {"status": "failed", "error": str(e)}
        
        # Load TTS
        try:
            from ..tts import TTSEngine
            self.models["tts"] = TTSEngine()
            results["tts"] = {"status": "loaded", "model": settings.tts_model}
        except Exception as e:
            logger.error(f"Failed to load TTS: {e}")
            results["tts"] = {"status": "failed", "error": str(e)}
        
        # Run dummy inference to warm up
        if "asr" in self.models:
            # TODO: Run dummy ASR
            pass
        
        if "llm" in self.models:
            # TODO: Run dummy LLM
            pass
        
        if "tts" in self.models:
            # TODO: Run dummy TTS
            pass
        
        logger.info(f"Model warmup complete: {results}")
        return results
    
    async def reload_model(self, model_type: str) -> Dict[str, Any]:
        """Reload a specific model."""
        if model_type not in ["asr", "llm", "tts"]:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Unload existing
        if model_type in self.models:
            del self.models[model_type]
        
        # Reload
        try:
            if model_type == "asr":
                from ..asr import ASREngine
                self.models["asr"] = ASREngine()
            elif model_type == "llm":
                from ..llm import LLMEngine
                self.models["llm"] = LLMEngine()
            elif model_type == "tts":
                from ..tts import TTSEngine
                self.models["tts"] = TTSEngine()
            
            return {"status": "reloaded", "model": getattr(settings, f"{model_type}_model")}
        
        except Exception as e:
            logger.error(f"Failed to reload {model_type}: {e}")
            return {"status": "failed", "error": str(e)}
    
    def get_loaded_models(self) -> Dict[str, bool]:
        """Get loaded model status."""
        return {
            "asr": "asr" in self.models,
            "llm": "llm" in self.models,
            "tts": "tts" in self.models
        }
    
    def get_model(self, model_type: str) -> Optional[Any]:
        """Get a loaded model."""
        return self.models.get(model_type)
