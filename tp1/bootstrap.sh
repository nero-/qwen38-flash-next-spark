#!/usr/bin/env bash
# Copy the single-Spark package, pull the pinned image, download and verify the
# step-5500 hybrid checkpoint, and install the pinned benchmark harness.
# Run on the Mac. This script does not start a server.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "$0")" && pwd)"
RESET_SELECTION=0
for arg in "$@"; do
  case "$arg" in
    --reset-selection) RESET_SELECTION=1 ;;
    *) echo 'Usage: bash tp1/bootstrap.sh [--reset-selection]' >&2; exit 2 ;;
  esac
done
cfg() { python3 "$HERE/config.py" "$1"; }
HOST="$(cfg ssh)" REMOTE_ROOT="$(cfg remote_root)" IMAGE="$(cfg image)" DEFAULT_PROFILE="$(cfg default_profile)"
BENCH_COMMIT=ccd9ad8ced7e387794391bfb0ac6d99b1f66ba6f
SSH_ARGS=(-o BatchMode=yes -o ConnectTimeout=10)
remote() { ssh "${SSH_ARGS[@]}" "$HOST" "$@"; }

remote "docker info >/dev/null"
if remote "docker inspect -f '{{.State.Running}}' qwen-tp1 2>/dev/null" | grep -qx true; then
  echo "Refusing bootstrap while qwen-tp1 is running on $HOST; stop it first." >&2
  exit 1
fi
if remote "docker inspect -f '{{.State.Running}}' qwen-tp2 2>/dev/null" | grep -qx true; then
  echo "Refusing bootstrap: $HOST is serving the TP2 deployment." >&2
  exit 1
fi

remote "mkdir -p ~/$REMOTE_ROOT/evidence ~/$REMOTE_ROOT/manifests ~/$REMOTE_ROOT/model-5500h"
rsync -a --exclude tests "$HERE/" "$HOST:$REMOTE_ROOT/"
rsync -a "$HERE/../tp2/manifests/model-5500h.sha256" "$HOST:$REMOTE_ROOT/manifests/"
if [[ $RESET_SELECTION == 1 ]]; then
  remote "printf '%s\\n' '$DEFAULT_PROFILE' > ~/$REMOTE_ROOT/selected-profile.txt"
else
  remote "test -e ~/$REMOTE_ROOT/selected-profile.txt || printf '%s\\n' '$DEFAULT_PROFILE' > ~/$REMOTE_ROOT/selected-profile.txt"
fi
remote "docker pull '$IMAGE'"

# Every file except README.md (the Hugging Face copy carries the model card) must match.
if ! remote "cd ~/$REMOTE_ROOT/model-5500h && grep -v '  ./README.md\$' ../manifests/model-5500h.sha256 | sha256sum --quiet -c - 2>/dev/null"; then
  remote "docker run --rm --network host --user \$(id -u):\$(id -g) --env HOME=\$HOME --env HF_TOKEN --mount type=bind,src=\$HOME/$REMOTE_ROOT,dst=\$HOME/$REMOTE_ROOT --entrypoint /usr/bin/python3 '$IMAGE' \$HOME/$REMOTE_ROOT/download_model.py"
  remote "cd ~/$REMOTE_ROOT/model-5500h && grep -v '  ./README.md\$' ../manifests/model-5500h.sha256 | sha256sum --quiet -c -"
fi

# Unmodified llm-inference-bench v0.6.2 in a small host venv (httpx + rich only).
remote "cd ~/$REMOTE_ROOT && { test -d llm-inference-bench || git clone -q https://github.com/local-inference-lab/llm-inference-bench.git; } \
  && git -C llm-inference-bench fetch -q origin && git -C llm-inference-bench checkout -q $BENCH_COMMIT \
  && { test -x .venv/bin/python || python3 -m venv .venv; } && .venv/bin/pip install -q httpx rich"
echo "Bootstrap complete: pinned image, verified step-5500 hybrid checkpoint and benchmark harness. No server was started."
