#!/usr/bin/env bash
# Read-only readiness check. From the Mac: bash tp1/preflight.sh (runs --local over SSH).
set -euo pipefail
HERE="$(cd -- "$(dirname -- "$0")" && pwd)"
get() { python3 "$HERE/config.py" "$1"; }
REMOTE_ROOT="$(get remote_root)"
if [[ "${1:-}" != "--local" ]]; then
  exec ssh -o BatchMode=yes -o ConnectTimeout=10 "$(get ssh)" "bash ~/$REMOTE_ROOT/preflight.sh --local"
fi

# Check what the selected profile will actually use: its image and every bind-mount source.
python3 "$HERE/launch.py" check | python3 -c '
import json, os, subprocess, sys
cmd = json.load(sys.stdin)
image = cmd[cmd.index("-m") - 1]
subprocess.run(["docker", "image", "inspect", image], check=True, stdout=subprocess.DEVNULL)
for i, arg in enumerate(cmd):
    if arg == "--mount" and cmd[i + 1].endswith(",readonly"):
        src = cmd[i + 1].split("src=", 1)[1].split(",", 1)[0]
        if not os.path.isfile(os.path.join(src, "config.json")):
            sys.exit(f"missing checkpoint or serving directory: {src}")
print("image and checkpoint present:", image.split("@")[-1][:19])
'
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
if docker inspect -f '{{.State.Running}}' qwen-tp2 2>/dev/null | grep -qx true; then
  echo "qwen-tp2 is running on this host; do not start qwen-tp1 alongside it." >&2
  exit 1
fi
echo "Preflight passed: the selected profile validates and its image and checkpoint exist."
