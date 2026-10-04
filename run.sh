#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Use `env -i` to clear the Hermes-injected PYTHONPATH which carries
# the incompatible global site-packages. Set only what we need.
env -u PYTHONPATH \
  HOME="$HOME" \
  PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  CUDA_VISIBLE_DEVICES=0 \
  "$HOME/venv/asr-streaming/bin/python3" -m uvicorn backend.main:app --host 0.0.0.0 --port 8740 --log-level info
