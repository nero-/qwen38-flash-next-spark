#!/usr/bin/env bash
# Copy the single-Spark package, pull the default profile's pinned image, download and
# verify its checkpoint (the step-5500 hybrid, or its NVFP4-CSF container for +csf
# profiles), and install the pinned benchmark harness.
# Run on the Mac. This script does not start a server.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "$0")" && pwd)"
RESET_SELECTION=0 ALL_MODELS=0
for arg in "$@"; do
  case "$arg" in
    --reset-selection) RESET_SELECTION=1 ;;
    --all-models) ALL_MODELS=1 ;;
    *) echo 'Usage: bash tp1/bootstrap.sh [--reset-selection] [--all-models]' >&2; exit 2 ;;
  esac
done
cfg() { python3 "$HERE/config.py" "$1"; }
HOST="$(cfg ssh)" REMOTE_ROOT="$(cfg remote_root)" DEFAULT_PROFILE="$(cfg default_profile)"
# The default profile's image and checkpoint (an experimental image modifier swaps the image).
read -r IMAGE NEEDS_CSF < <(python3 - "$HERE" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from pathlib import Path
from config import load_config, parse_profile
c = load_config(Path(sys.argv[1]))
images = c.get("experimental_images", {})
p = parse_profile(c["default_profile"], images)
print(images[p["image"]] if p["image"] else c["image"], int("csf" in p["flags"]))
PY
)
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

remote "mkdir -p ~/$REMOTE_ROOT/evidence ~/$REMOTE_ROOT/manifests ~/$REMOTE_ROOT/model-5500h ~/$REMOTE_ROOT/model-csf"
rsync -a --exclude tests "$HERE/" "$HOST:$REMOTE_ROOT/"
rsync -a "$HERE/../tp2/manifests/model-5500h.sha256" "$HOST:$REMOTE_ROOT/manifests/"
if [[ $RESET_SELECTION == 1 ]]; then
  remote "printf '%s\\n' '$DEFAULT_PROFILE' > ~/$REMOTE_ROOT/selected-profile.txt"
else
  remote "test -e ~/$REMOTE_ROOT/selected-profile.txt || printf '%s\\n' '$DEFAULT_PROFILE' > ~/$REMOTE_ROOT/selected-profile.txt"
fi
if [[ "$IMAGE" == sha256:* ]]; then
  # A locally built image (no registry digest): it cannot be pulled, only rebuilt.
  remote "docker image inspect '$IMAGE' >/dev/null 2>&1" || {
    echo "The default profile's image $IMAGE is a local build that is missing on $HOST." >&2
    echo "Stop qwen-tp1, run 'bash ~/$REMOTE_ROOT/images/build-kk-csf.sh' there (~95 min)," >&2
    echo "then set experimental_images in tp1/node-config.json to the printed image ID and rerun." >&2
    exit 1
  }
else
  remote "docker pull '$IMAGE'"
fi

download() {
  remote "docker run --rm --network host --user \$(id -u):\$(id -g) --env HOME=\$HOME --env HF_TOKEN --mount type=bind,src=\$HOME/$REMOTE_ROOT,dst=\$HOME/$REMOTE_ROOT --entrypoint /usr/bin/python3 '$IMAGE' \$HOME/$REMOTE_ROOT/download_model.py $1"
}
if [[ $NEEDS_CSF == 0 || $ALL_MODELS == 1 ]]; then
  # Every file except README.md (the Hugging Face copy carries the model card) must match.
  if ! remote "cd ~/$REMOTE_ROOT/model-5500h && grep -v '  ./README.md\$' ../manifests/model-5500h.sha256 | sha256sum --quiet -c - 2>/dev/null"; then
    download hybrid
    remote "cd ~/$REMOTE_ROOT/model-5500h && grep -v '  ./README.md\$' ../manifests/model-5500h.sha256 | sha256sum --quiet -c -"
  fi
fi
if [[ $NEEDS_CSF == 1 || $ALL_MODELS == 1 ]]; then
  # SHA256SUMS is pinned in node-config.json; every file must match it.
  remote "cd ~/$REMOTE_ROOT && python3 prepare_csf.py verify 2>/dev/null" || {
    download csf
    remote "cd ~/$REMOTE_ROOT && python3 prepare_csf.py verify"
  }
  remote "cd ~/$REMOTE_ROOT && python3 prepare_csf.py serve"
fi

# Unmodified llm-inference-bench v0.6.2 in a small host venv (httpx + rich only).
remote "cd ~/$REMOTE_ROOT && { test -d llm-inference-bench || git clone -q https://github.com/local-inference-lab/llm-inference-bench.git; } \
  && git -C llm-inference-bench fetch -q origin && git -C llm-inference-bench checkout -q $BENCH_COMMIT \
  && { test -x .venv/bin/python || python3 -m venv .venv; } && .venv/bin/pip install -q httpx rich"
echo "Bootstrap complete: pinned image, verified checkpoint(s) and benchmark harness. No server was started."
