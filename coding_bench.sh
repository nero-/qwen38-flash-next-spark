#!/usr/bin/env bash
# Coding-speed benchmark: long Python/SQL code-generation prompt at C=1, thinking off,
# temperature 0. Reports single-stream decode tok/s (max coding speed at C=1) and TTFT.
set -euo pipefail
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
OUT="${OUT:-$HOME/coding_speed.json}"

PROMPT='Write a production-quality Python function `analyze_sales(rows)` that takes a list of dicts with keys date (ISO string), region (str), product (str), units (int), revenue (float). Return a dict with: (1) revenue_by_region sorted desc, (2) top_3_products by total units with ties broken by name asc, (3) month_over_month growth of total revenue as percentages rounded to 2 decimals. Include type hints, docstring, input validation raising ValueError on bad rows, and a small usage example under if __name__ == "__main__". Write only code, no prose.'

curl -s "http://$HOST:$PORT/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "$(python3 - "$PROMPT" <<'PY'
import json, sys
print(json.dumps({
    "model": "qwen3.8-flash-next-4p89bpw",
    "messages": [{"role": "user", "content": sys.argv[1]}],
    "max_tokens": 2048,
    "temperature": 0,
    "stream": True,
    "chat_template_kwargs": {"enable_thinking": False},
}))
PY
)" | python3 - "$OUT" <<'PY'
import json, sys, time
out_path = sys.argv[1]
t0 = None; t_first = None; t_last = None; n = 0
for line in sys.stdin:
    line = line.strip()
    if not line.startswith("data: ") or line == "data: [DONE]":
        continue
    d = json.loads(line[6:])
    ch = d["choices"][0]
    delta = ch.get("delta") or {}
    if delta.get("content"):
        now = time.monotonic()
        if t0 is None:
            t0 = now
        if t_first is None:
            t_first = now
        t_last = now
        n += len(delta["content"])
    if d.get("usage"):
        usage = d["usage"]
gen_s = (t_last - t_first) if (t_first and t_last and t_last > t_first) else 0
ttft = (t_first - t0) if (t_first and t0) else 0
result = {
    "decode_tokens": n,
    "ttft_s": round(ttft, 3),
    "generation_s": round(gen_s, 3),
    "decode_tok_s": round(n / gen_s, 2) if gen_s else None,
}
print(json.dumps(result, indent=2))
with open(out_path, "w") as fh:
    json.dump(result, fh, indent=2)
PY
