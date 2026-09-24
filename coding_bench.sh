#!/usr/bin/env bash
# Official local-inference-lab/llm-inference-bench Coding Peak mode.
# A short C=1 decode cell precedes the built-in sequential coding test.
# Coding Peak uses the upstream Sieve prompt and model sampling/thinking defaults.
set -euo pipefail
BENCH_ROOT="${BENCH_ROOT:-$HOME/projects/llm-inference-bench}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/venvs/qwen38/bin/python}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-qwen3.8-flash-next-4p89bpw}"
OUT="${OUT:-$HOME/coding_peak_$(date +%Y%m%d_%H%M%S).json}"
[[ -f "$BENCH_ROOT/llm_decode_bench.py" ]] || { echo "Missing benchmark checkout: $BENCH_ROOT" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python environment: $PYTHON_BIN" >&2; exit 1; }
extra=()
if [[ -n "${CODING_TEMPERATURE:-}" ]]; then
  extra+=(--coding-peak-temperature "$CODING_TEMPERATURE")
fi
exec "$PYTHON_BIN" "$BENCH_ROOT/llm_decode_bench.py" \
  --host "$HOST" --port "$PORT" --model "$MODEL" \
  --no-hw-monitor --display-mode plain --no-resume \
  --skip-prefill --contexts 0 --concurrency 1 --duration 15 \
  --max-tokens 2048 --coding-peak \
  --coding-peak-runs "${CODING_RUNS:-5}" \
  --coding-peak-max-tokens "${CODING_MAX_TOKENS:-2000}" \
  --output "$OUT" "${extra[@]}" "$@"
