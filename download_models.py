#!/usr/bin/env python3
"""
Download Whisper models for offline use.
This helps avoid network issues during processing.
"""

import whisper
import sys
import os
from pathlib import Path

def download_model(model_name: str):
    """Download a specific Whisper model."""
    print(f"Downloading Whisper model '{model_name}'...")
    try:
        # This will download and cache the model
        model = whisper.load_model(model_name)
        print(f"✓ Model '{model_name}' downloaded successfully")
        
        # Show where model is cached
        cache_dir = Path.home() / ".cache" / "whisper"
        print(f"  Cached in: {cache_dir}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to download model '{model_name}': {e}")
        return False

def main():
    """Download commonly used models."""
    models_to_download = ["tiny", "base"]
    
    if len(sys.argv) > 1:
        # Download specific models from command line
        models_to_download = sys.argv[1:]
    
    print("Whisper Model Downloader")
    print("=" * 40)
    print(f"Models to download: {', '.join(models_to_download)}")
    print()
    
    success_count = 0
    for model_name in models_to_download:
        if download_model(model_name):
            success_count += 1
        print()
    
    print("=" * 40)
    print(f"Downloaded {success_count}/{len(models_to_download)} models successfully")
    
    if success_count < len(models_to_download):
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Try using a VPN if the download server is blocked")
        print("3. Manually download from: https://github.com/openai/whisper#available-models-and-languages")
        sys.exit(1)

if __name__ == "__main__":
    main()