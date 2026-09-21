#!/usr/bin/env bash
# One-shot install of the Qwen3.8-Flash-Next NVFP4 (QAD) serving stack.
# Ubuntu 24.04 aarch64, CUDA 13.x, ~150 GiB free disk.
set -euo pipefail

VLLM_BRANCH=dev/karmic-kraken

echo "==> System deps (python dev headers, io_uring)"
sudo apt-get install -y python3.12-dev liburing-dev pkg-config

echo "==> uv"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
UV="$HOME/.local/bin/uv"

echo "==> Checkouts"
mkdir -p ~/projects ~/venvs ~/models
for repo in vllm b12x; do
  [ -d ~/projects/$repo ] || git clone --filter=blob:none \
    https://github.com/local-inference-lab/$repo.git ~/projects/$repo
done
git -C ~/projects/vllm fetch origin $VLLM_BRANCH
git -C ~/projects/vllm checkout $VLLM_BRANCH

echo "==> venv + torch 2.13.0+cu132"
$UV venv ~/venvs/qwen38 --python 3.12
$UV pip install --python ~/venvs/qwen38/bin/python \
  torch==2.13.0 --index-url https://download.pytorch.org/whl/cu132

echo "==> vLLM (source build, 30-60 min)"
$UV pip install --python ~/venvs/qwen38/bin/python \
  setuptools_rust setuptools_scm ninja wheel jinja2
$UV pip install --python ~/venvs/qwen38/bin/python \
  -r ~/projects/vllm/requirements/common.txt
$UV pip install --python ~/venvs/qwen38/bin/python \
  -e ~/projects/vllm --no-build-isolation --no-deps

echo "==> b12x (source, required: PyPI wheel lacks SM121 dense-activation changes)"
$UV pip install --python ~/venvs/qwen38/bin/python --no-deps -e ~/projects/b12x

echo "==> Checkpoint (98.6 GiB)"
$UV venv ~/venvs/hfdl --python 3.12
$UV pip install --python ~/venvs/hfdl/bin/python "huggingface_hub[hf_transfer]"
HF_HUB_ENABLE_HF_TRANSFER=1 ~/venvs/hfdl/bin/hf download \
  local-inference-lab/Qwen3.8-Flash-Next-NVFP4 \
  --local-dir ~/models/Qwen3.8-Flash-Next-NVFP4 --max-workers 6

echo "Done. Run: bash serve.sh"
