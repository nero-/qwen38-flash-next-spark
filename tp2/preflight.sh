#!/usr/bin/env bash
# Mac entry point. --local is the read-only rank-0 check run over SSH.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "$0")" && pwd)"
get() { python3 "$HERE/config.py" "$1"; }
REMOTE_ROOT="$(get remote_root)" IMAGE="$(get image)"
HEAD="$(get ranks.0.ssh)" WORKER="$(get ranks.1.peer_ssh)"
SSH_ARGS=(-o BatchMode=yes -o ConnectTimeout=10)
if [[ "${1:-}" != "--local" ]]; then
  exec ssh "${SSH_ARGS[@]}" "$HEAD" "bash ~/$REMOTE_ROOT/preflight.sh --local"
fi

test -f "$HOME/$REMOTE_ROOT/model/config.json"
docker image inspect "$IMAGE" >/dev/null
python3 "$HERE/launch.py" check --rank 0 >/dev/null
ssh "${SSH_ARGS[@]}" "$WORKER" \
  "test -f \$HOME/$REMOTE_ROOT/model/config.json && docker image inspect '$IMAGE' >/dev/null && python3 \$HOME/$REMOTE_ROOT/launch.py check --rank 1 >/dev/null"
echo "Preflight passed: selected profile and cable mapping validate on both ranks; pinned image and model files exist."
