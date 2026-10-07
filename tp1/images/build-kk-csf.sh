#!/usr/bin/env bash
# Build the "kk1007" serving image on the Spark: eugr/spark-vllm-docker's B12X image with
# vLLM, b12x and FlashInfer pinned to the commits validated for NVFP4-CSF. Run on the Spark.
#
# This is the nightly build (--exp-b12x --rebuild-vllm) with three local edits:
#   - InstantTensor pinned to 0.2.0: 0.2.1 (2026-10-06) breaks eugr's InstantTensor patch step
#   - vLLM pinned to the KK commit below instead of the dev/karmic-kraken branch head
#   - b12x pinned to the commit below instead of master (full clone + checkout)
# FlashInfer is pinned to nightly-20261006's commit; eugr's prebuilt FlashInfer wheel release
# was unavailable, so it is compiled. Expect ~95 minutes and most of the Spark's memory:
# stop qwen-tp1 first. A rebuilt image gets a new image ID; put it in node-config.json.
set -euo pipefail
EUGR_COMMIT=73f01ceea6369b8d4451462f094c20cb10559c11
VLLM_COMMIT=e07345714ba82bd32ab88b963223b37d2338ca5f
B12X_COMMIT=2cc7f66aaef76a9c7cbb660a1efb865cf3f26c88
FLASHINFER_COMMIT=f8d3729e9c85d20afdf4a1e672771cdb157e0e33
TAG=qwen-tp1-kk:csf
DIR="${EUGR_DIR:-$HOME/builds/eugr-spark-vllm-docker}"

if [[ "$(docker inspect -f '{{.State.Running}}' qwen-tp1 2>/dev/null)" == true ]]; then
  echo "Stop qwen-tp1 first: the vLLM and FlashInfer compiles need the memory it holds." >&2
  exit 1
fi
[[ -d "$DIR/.git" ]] || git clone -q https://github.com/eugr/spark-vllm-docker.git "$DIR"
cd "$DIR"
git fetch -q origin
git checkout -q --force "$EUGR_COMMIT"

edit() {  # edit FILE OLD NEW: replace exactly one occurrence, or fail
  python3 - "$@" <<'PY'
import sys
from pathlib import Path
path, old, new = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
text = path.read_text()
if text.count(old) != 1:
    sys.exit(f"{path}: expected exactly one {old!r}, found {text.count(old)}")
path.write_text(text.replace(old, new))
PY
}
edit Dockerfile "uv pip install ray[default] fastsafetensors instanttensor \\" \
                "uv pip install ray[default] fastsafetensors instanttensor==0.2.0 \\"
edit Dockerfile 'git clone --depth 1 --branch "$B12X_REF" "$B12X_REPO" /tmp/b12x-source && \' \
                'git clone "$B12X_REPO" /tmp/b12x-source && git -C /tmp/b12x-source checkout --detach "$B12X_REF" && \'
edit build-and-copy.sh 'EXP_B12X_VLLM_REF="dev/karmic-kraken"' "EXP_B12X_VLLM_REF=\"$VLLM_COMMIT\""
edit build-and-copy.sh 'B12X_PACKAGE_REF="master"' "B12X_PACKAGE_REF=\"$B12X_COMMIT\""

./build-and-copy.sh --exp-b12x --rebuild-vllm --rebuild-flashinfer \
  --flashinfer-ref "$FLASHINFER_COMMIT" -t "$TAG" --full-log
git checkout -q -- Dockerfile build-and-copy.sh

# The image must carry the pinned commits and the NVFP4-CSF reader.
docker run --rm --entrypoint bash "$TAG" -c "
  set -e
  grep -q '^vllm_commit: $VLLM_COMMIT' /workspace/build-metadata.yaml
  grep -q '^flashinfer_commit: $FLASHINFER_COMMIT' /workspace/build-metadata.yaml
  test \"\$(cat /workspace/b12x-source-commit)\" = $B12X_COMMIT
  python3 -c 'import vllm.model_executor.layers.quantization.nvfp4_csf, vllm.model_executor.model_loader.nvfp4_csf_loader'
  python3 -c 'from b12x.moe import fused_moe; assert hasattr(fused_moe, \"expand_scales\")'
"
echo "Built $TAG: $(docker image inspect "$TAG" --format '{{.Id}}')"
echo "Set experimental_images.kk1007 in node-config.json to that ID if it changed."
