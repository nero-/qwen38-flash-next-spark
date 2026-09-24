#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next NVFP4 (QAD) on ONE DGX Spark, TP=1.
# Wraps the branch's serve-qwen38-flash-next-nvfp4-tp1.sh with local paths.
set -euo pipefail

VLLM_ROOT="${VLLM_ROOT:-$HOME/projects/vllm}"
B12X_ROOT="${B12X_ROOT:-$HOME/projects/b12x}"
MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen3.8-Flash-Next-NVFP4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.3}"
LOAD_FORMAT="${LOAD_FORMAT:-b12x}"
PLE_MODE="${PLE_MODE:-${VLLM_PLE_TABLE_MEMORY:-disk}}"
case "$PLE_MODE" in
  resident)
    ple_env=(-u VLLM_PLE_TABLE_MEMORY VLLM_PLE_CPU_OFFLOAD=0)
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
    KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-4294967296}"
    ;;
  disk)
    ple_env=(VLLM_PLE_TABLE_MEMORY=disk)
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
    KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-21474836480}"
    ;;
  *) echo "PLE_MODE must be resident or disk" >&2; exit 2 ;;
esac

check=0
args=()
for arg in "$@"; do
  if [[ "$arg" == --check ]]; then
    check=1
  else
    args+=("$arg")
  fi
done

[ -f "$MODEL_PATH/config.json" ] || { echo "Checkpoint missing: $MODEL_PATH" >&2; exit 1; }
[ -f "$VLLM_ROOT/serve-qwen38-flash-next-nvfp4-tp1.sh" ] || { echo "vLLM checkout missing: $VLLM_ROOT" >&2; exit 1; }

cd "$VLLM_ROOT"
export PYTHON_BIN="$VLLM_ROOT/.venv/bin/python"
[ -x "$PYTHON_BIN" ] || PYTHON_BIN="$HOME/venvs/qwen38/bin/python"

command=(env \
  "${ple_env[@]}" \
  B12X_POLICY_MODE=auto \
  CUDA_HOME="$CUDA_HOME" \
  TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" \
  PATH="$(dirname "$PYTHON_BIN"):$CUDA_HOME/bin:$PATH" \
  OMP_NUM_THREADS="$OMP_NUM_THREADS" \
  MODEL_PATH="$MODEL_PATH" \
  HOST="$HOST" PORT="$PORT" \
  MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  MAX_NUM_SEQS="$MAX_NUM_SEQS" \
  MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" \
  KV_CACHE_MEMORY_BYTES="$KV_CACHE_MEMORY_BYTES" \
  NUM_SPECULATIVE_TOKENS="$NUM_SPECULATIVE_TOKENS" \
  bash "$VLLM_ROOT/serve-qwen38-flash-next-nvfp4-tp1.sh" \
  --load-format "$LOAD_FORMAT" \
  --mamba-ssm-cache-dtype bfloat16 \
  --gdn-prefill-backend flashinfer \
  --gdn-decode-kernel cuda \
  --limit-mm-per-prompt '{}' \
  "${args[@]}")

if ((check)); then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
exec "${command[@]}"
