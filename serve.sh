#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next NVFP4 (QAD) on ONE DGX Spark, TP=1.
# Wraps the branch's serve-qwen38-flash-next-nvfp4-tp1.sh with local paths.
set -euo pipefail

VLLM_ROOT="${VLLM_ROOT:-$HOME/projects/vllm}"
B12X_ROOT="${B12X_ROOT:-$HOME/projects/b12x}"
MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen3.8-Flash-Next-NVFP4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-1610612736}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"

[ -f "$MODEL_PATH/config.json" ] || { echo "Checkpoint missing: $MODEL_PATH" >&2; exit 1; }
[ -f "$VLLM_ROOT/serve-qwen38-flash-next-nvfp4-tp1.sh" ] || { echo "vLLM checkout missing: $VLLM_ROOT" >&2; exit 1; }

cd "$VLLM_ROOT"
export PYTHON_BIN="$VLLM_ROOT/.venv/bin/python"
[ -x "$PYTHON_BIN" ] || PYTHON_BIN="$HOME/venvs/qwen38/bin/python"

exec env \
  B12X_POLICY_MODE=auto \
  MODEL_PATH="$MODEL_PATH" \
  HOST="$HOST" PORT="$PORT" \
  MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  MAX_NUM_SEQS="$MAX_NUM_SEQS" \
  MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" \
  KV_CACHE_MEMORY_BYTES="$KV_CACHE_MEMORY_BYTES" \
  NUM_SPECULATIVE_TOKENS="$NUM_SPECULATIVE_TOKENS" \
  bash "$VLLM_ROOT/serve-qwen38-flash-next-nvfp4-tp1.sh" "$@"
