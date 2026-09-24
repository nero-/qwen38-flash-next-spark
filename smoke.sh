#!/usr/bin/env bash
# Smoke test: /health then one OpenAI completion.
set -euo pipefail
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/venvs/qwen38/bin/python}"

for i in $(seq 1 120); do
  if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then break; fi
  sleep 10
done
curl -fsS "http://$HOST:$PORT/health" >/dev/null || { echo "server not healthy" >&2; exit 1; }

curl -fsS "http://$HOST:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-flash-next-4p89bpw","messages":[{"role":"user","content":"Say OK and nothing else."}],"max_tokens":16,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
  | "$PYTHON_BIN" -c 'import json,sys; r=json.load(sys.stdin); text=r["choices"][0]["message"]["content"]; assert text and text.strip()=="OK", r; print(text)'
