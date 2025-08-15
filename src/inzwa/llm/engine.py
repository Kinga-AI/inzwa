"""LLM engine implementation with pluggable backends."""

import asyncio
from typing import AsyncIterator, List, Dict, Any, Optional
from ..config import settings
from ..models import TokenChunk
from ..telemetry import get_logger, llm_ttfw_histogram

logger = get_logger(__name__)


class LLMEngine:
    """LLM engine with support for vLLM and llama-cpp."""
    
    def __init__(self):
        self.engine_type = settings.llm_engine
        self.model_name = settings.llm_model
        self.device = settings.llm_device
        self.max_tokens = settings.llm_max_tokens
        self.temperature = settings.llm_temperature
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the LLM based on configuration."""
        if self.engine_type == "llama-cpp":
            self._load_llama_cpp()
        elif self.engine_type == "vllm":
            self._load_vllm()
        else:
            raise ValueError(f"Unknown LLM engine: {self.engine_type}")
    
    def _load_llama_cpp(self):
        """Load llama-cpp model."""
        try:
            from llama_cpp import Llama
            
            model_path = settings.llm_model_path or f"models/{self.model_name}.gguf"
            
            self.model = Llama(
                model_path=model_path,
                n_ctx=settings.llm_context_size,
                n_gpu_layers=settings.llm_gpu_layers if self.device == "cuda" else 0,
                verbose=False
            )
            logger.info(f"Loaded llama-cpp model: {model_path}")
        except ImportError:
            logger.error("llama-cpp-python not installed")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to load llama-cpp model: {e}")
            self.model = None
    
    def _load_vllm(self):
        """Load vLLM model."""
        try:
            from vllm import AsyncLLMEngine, SamplingParams
            
            # TODO: Implement vLLM loading
            logger.warning("vLLM backend not fully implemented")
            self.model = None
        except ImportError:
            logger.error("vLLM not installed")
            self.model = None
    
    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> AsyncIterator[TokenChunk]:
        """Stream generation with partial tokens."""
        
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature
        
        if self.model is None:
            # Placeholder implementation
            yield TokenChunk(
                token="[LLM not loaded] ",
                logprob=None,
                is_final=True
            )
            return
        
        # Format messages into prompt
        prompt = self._format_prompt(messages)
        
        if self.engine_type == "llama-cpp":
            async for token in self._generate_llama_cpp(prompt, max_tokens, temperature):
                yield token
        elif self.engine_type == "vllm":
            async for token in self._generate_vllm(prompt, max_tokens, temperature):
                yield token
    
    def _format_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into a prompt string."""
        # Simple format for now, can be customized per model
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        prompt += "Assistant: "
        return prompt
    
    async def _generate_llama_cpp(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> AsyncIterator[TokenChunk]:
        """Generate with llama-cpp."""
        if not self.model:
            return
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        # Track time to first token
        first_token = True
        
        # Generate tokens
        stream = await loop.run_in_executor(
            None,
            lambda: self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                stop=["User:", "\n\n"]
            )
        )
        
        for output in stream:
            token = output["choices"][0]["text"]
            
            if first_token:
                # Record TTFW metric
                first_token = False
            
            yield TokenChunk(
                token=token,
                logprob=output["choices"][0].get("logprobs"),
                is_final=False
            )
        
        # Final token
        yield TokenChunk(token="", logprob=None, is_final=True)
    
    async def _generate_vllm(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> AsyncIterator[TokenChunk]:
        """Generate with vLLM."""
        # TODO: Implement vLLM streaming
        yield TokenChunk(
            token="[vLLM not implemented]",
            logprob=None,
            is_final=True
        )
    
    async def generate_batch(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """Batch generation of complete response."""
        
        full_text = ""
        async for chunk in self.generate_stream(messages, max_tokens, temperature):
            if not chunk.is_final:
                full_text += chunk.token
        
        return full_text
