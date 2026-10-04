#!/usr/bin/env python3
"""
Wrapper that cleans sys.path to avoid the global Hermes venv's incompatible
huggingface_hub, then launches the ASR streaming server.
"""
import sys
import os

# Remove all paths from the global Hermes venv
_hermes_hack_paths = [p for p in sys.path if "hermes" in p.lower() or "hermes" in p.split("/")[-1]]
for p in _hermes_hack_paths:
    if "installs" in p:
        sys.path.remove(p)

# Ensure the project's own packages are first
project_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_path)

# Now run uvicorn
import uvicorn
from backend.main import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8740, log_level="info")
