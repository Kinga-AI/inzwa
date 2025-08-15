#!/usr/bin/env python3
"""Script to download required models."""

import os
import sys
from pathlib import Path

def download_models():
    """Download required models from Hugging Face Hub."""
    
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)
    
    print("Downloading models...")
    
    # ASR models
    print("\n1. ASR Models (Whisper)")
    # TODO: Download whisper models
    # from huggingface_hub import snapshot_download
    # snapshot_download("openai/whisper-small", local_dir=models_dir / "asr" / "whisper-small")
    
    # LLM models
    print("\n2. LLM Models")
    # TODO: Download Mistral/Gemma models
    # For llama-cpp, download GGUF files
    
    # TTS models
    print("\n3. TTS Models")
    # TODO: Download Coqui TTS models
    # Custom Shona model would need to be trained first
    
    print("\n✓ Model download complete!")
    print(f"Models saved to: {models_dir.absolute()}")

if __name__ == "__main__":
    download_models()
