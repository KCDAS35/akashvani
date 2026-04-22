#!/usr/bin/env bash
# Akash Vani — Voice from the Sky
# Usage: ./akashvani.sh [model]
# Models: tiny.en (fastest), base.en (default), small.en (more accurate)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export WHISPER_MODEL="${1:-base.en}"
export PATH="$HOME/.local/bin:$PATH"

exec python3 "$SCRIPT_DIR/akashvani.py"
