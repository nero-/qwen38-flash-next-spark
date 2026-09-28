#!/usr/bin/env bash
# Read-only readiness check. From the Mac: bash tp1/preflight.sh (runs --local over SSH).
set -euo pipefail
HERE="$(cd -- "$(dirname -- "$0")" && pwd)"
get() { python3 "$HERE/config.py" "$1"; }
REMOTE_ROOT="$(get remote_root)" IMAGE="$(get image)"
if [[ "${1:-}" != "--local" ]]; then
  exec ssh -o BatchMode=yes -o ConnectTimeout=10 "$(get ssh)" "bash ~/$REMOTE_ROOT/preflight.sh --local"
fi

test -f "$HOME/$REMOTE_ROOT/model-5500h/HYBRID.json"
docker image inspect "$IMAGE" >/dev/null
python3 "$HERE/launch.py" check >/dev/null
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
if docker inspect -f '{{.State.Running}}' qwen-tp2 2>/dev/null | grep -qx true; then
  echo "qwen-tp2 is running on this host; do not start qwen-tp1 alongside it." >&2
  exit 1
fi
echo "Preflight passed: selected profile validates; pinned image and step-5500 hybrid exist."
