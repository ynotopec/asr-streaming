#!/usr/bin/env python3
"""
Wrapper that launches the audio-to-audio translator server
using the Python 3.12 venv (needed for Nemotron ASR / Qwen3-TTS).
"""
import sys
import os

# Clean Hermes-injected paths
sys.path = [p for p in sys.path if not p.startswith("/home/ai-agent/.hermes/installs")]

# Ensure backend module is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

import uvicorn
from backend.main import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8741, log_level="info")