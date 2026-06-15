#!/usr/bin/env bash
# Akash Vani — Voice from the Sky
# Usage: ./akashvani.sh [model]
# Models: tiny.en (fastest), base.en (default), small.en (more accurate)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export WHISPER_MODEL="${1:-base.en}"
export PATH="$HOME/.local/bin:$PATH"

# Mutter (Wayland) generates a per-session XAUTHORITY at /run/user/$UID/.mutter-Xwaylandauth.*
# The systemd unit's static XAUTHORITY=~/.Xauthority goes stale across logins; pick the live one.
export DISPLAY="${DISPLAY:-:0}"
LIVE_XAUTH=$(ls -t /run/user/"$(id -u)"/.mutter-Xwaylandauth.* 2>/dev/null | head -1)
[ -n "$LIVE_XAUTH" ] && export XAUTHORITY="$LIVE_XAUTH"

exec python3 "$SCRIPT_DIR/akashvani.py"
