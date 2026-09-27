#!/usr/bin/env bash
# Copy the deployment package to both Sparks, pull the pinned image, and verify
# independently downloaded snapshots. This script does not start a server.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "$0")" && pwd)"
CONFIG="$HERE/cluster-config.json"
RESET_SELECTION=0 BUILD_HYBRID=0
for arg in "$@"; do
  case "$arg" in
    --reset-selection) RESET_SELECTION=1 ;;
    --build-hybrid) BUILD_HYBRID=1 ;;
    *) echo 'Usage: bash tp2/bootstrap.sh [--reset-selection] [--build-hybrid]' >&2; exit 2 ;;
  esac
done
cfg() { python3 "$HERE/config.py" "$1"; }
REMOTE_ROOT="$(cfg remote_root)" IMAGE="$(cfg image)"
HEAD="$(cfg ranks.0.ssh)" WORKER="$(cfg ranks.1.ssh)"
DEFAULT_CABLES="$(cfg default_cables)" DEFAULT_PROFILE="$(cfg default_profile)"
SSH_ARGS=(-o BatchMode=yes -o ConnectTimeout=10)

for host in "$HEAD" "$WORKER"; do
  # Do not alter a running deployment or replace its selected mode on rerun.
  ssh "${SSH_ARGS[@]}" "$host" "docker info >/dev/null"
  if ssh "${SSH_ARGS[@]}" "$host" "docker inspect -f '{{.State.Running}}' qwen-tp2 2>/dev/null" | grep -qx true; then
    echo "Refusing bootstrap while qwen-tp2 is running on $host; stop both ranks first." >&2
    exit 1
  fi
done

for host in "$HEAD" "$WORKER"; do
  ssh "${SSH_ARGS[@]}" "$host" "mkdir -p ~/$REMOTE_ROOT/cache ~/$REMOTE_ROOT/evidence ~/$REMOTE_ROOT/optimization ~/$REMOTE_ROOT/model ~/.cache/huggingface"
  rsync -a "$HERE/" "$host:~/$REMOTE_ROOT/"
  if [[ $RESET_SELECTION == 1 ]]; then
    ssh "${SSH_ARGS[@]}" "$host" "printf '%s\\n' '$DEFAULT_CABLES' > ~/$REMOTE_ROOT/selected-cables.txt; printf '%s\\n' '$DEFAULT_PROFILE' > ~/$REMOTE_ROOT/selected-profile.txt"
  else
    ssh "${SSH_ARGS[@]}" "$host" "test -e ~/$REMOTE_ROOT/selected-cables.txt || printf '%s\\n' '$DEFAULT_CABLES' > ~/$REMOTE_ROOT/selected-cables.txt; test -e ~/$REMOTE_ROOT/selected-profile.txt || printf '%s\\n' '$DEFAULT_PROFILE' > ~/$REMOTE_ROOT/selected-profile.txt"
  fi
  ssh "${SSH_ARGS[@]}" "$host" "docker pull '$IMAGE'"
done

for host in "$HEAD" "$WORKER"; do
  ssh "${SSH_ARGS[@]}" "$host" \
    "docker run --rm --network host --user \$(id -u):\$(id -g) --env HOME=\$HOME --env HF_TOKEN --mount type=bind,src=\$HOME/$REMOTE_ROOT,dst=\$HOME/$REMOTE_ROOT --mount type=bind,src=\$HOME/.cache/huggingface,dst=\$HOME/.cache/huggingface,readonly --entrypoint /usr/bin/python3 '$IMAGE' \$HOME/$REMOTE_ROOT/download_model.py"
  ssh "${SSH_ARGS[@]}" "$host" "python3 ~/$REMOTE_ROOT/model_manifest.py write"
done

# Default checkpoint: the step-5500 QAD hybrid (trained step-5500 tensors with main's
# MXFP8 attention and NVFP4 MTP experts), downloaded from its pinned Hugging Face
# revision. --build-hybrid instead downloads the step-5500 trunk and rebuilds it with
# optimization/build_hybrid.py. Either way every file except README.md (the HF copy
# carries the model card) must match the committed manifest.
for host in "$HEAD" "$WORKER"; do
  if [[ $BUILD_HYBRID == 1 ]]; then
    ssh "${SSH_ARGS[@]}" "$host" \
      "docker run --rm --network host --user \$(id -u):\$(id -g) --env HOME=\$HOME --env HF_TOKEN --mount type=bind,src=\$HOME/$REMOTE_ROOT,dst=\$HOME/$REMOTE_ROOT --entrypoint /usr/bin/python3 '$IMAGE' \$HOME/$REMOTE_ROOT/download_model.py trunk"
    ssh "${SSH_ARGS[@]}" "$host" "cd ~/$REMOTE_ROOT/model-5500 && sha256sum --quiet -c ../manifests/model-5500.sha256"
    ssh "${SSH_ARGS[@]}" "$host" \
      "test -d ~/$REMOTE_ROOT/model-5500h || docker run --rm --user \$(id -u):\$(id -g) --env HOME=\$HOME --mount type=bind,src=\$HOME/$REMOTE_ROOT,dst=\$HOME/$REMOTE_ROOT --entrypoint bash '$IMAGE' -c 'python3 \$HOME/$REMOTE_ROOT/optimization/build_hybrid.py verify && python3 \$HOME/$REMOTE_ROOT/optimization/build_hybrid.py build'"
  else
    ssh "${SSH_ARGS[@]}" "$host" \
      "test -d ~/$REMOTE_ROOT/model-5500h || docker run --rm --network host --user \$(id -u):\$(id -g) --env HOME=\$HOME --env HF_TOKEN --mount type=bind,src=\$HOME/$REMOTE_ROOT,dst=\$HOME/$REMOTE_ROOT --entrypoint /usr/bin/python3 '$IMAGE' \$HOME/$REMOTE_ROOT/download_model.py hybrid"
  fi
  ssh "${SSH_ARGS[@]}" "$host" "cd ~/$REMOTE_ROOT/model-5500h && grep -v '  ./README.md\$' ../manifests/model-5500h.sha256 | sha256sum --quiet -c -"
done

# OpenSSH streams the checksum manifest through the Mac; rsync cannot copy
# directly between two remote endpoints.
ssh "${SSH_ARGS[@]}" "$HEAD" "cat ~/$REMOTE_ROOT/model.sha256.json" | \
  ssh "${SSH_ARGS[@]}" "$WORKER" "cat > ~/$REMOTE_ROOT/model.sha256.json"
ssh "${SSH_ARGS[@]}" "$WORKER" "python3 ~/$REMOTE_ROOT/model_manifest.py check"
echo "Bootstrap complete. Pair has the pinned image and matching model snapshot. No server was started."
