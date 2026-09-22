# Operations Guide — Qwen3.8-Flash-Next NVFP4 (QAD) on spark-r0

Day-to-day commands for the TP1 deployment. Install/build details are in
`README.md`; benchmark numbers in `BENCHMARK.md`; the two-Spark path in
`TP2_MIGRATION.md`.

## Model name (for API clients)

```
qwen3.8-flash-next-4p89bpw
```

The API is OpenAI-compatible at `http://<spark-ip>:8000/v1` — chat
completions, completions, streaming, tool calling (`qwen3_xml` parser),
vision (images + video), reasoning parser `qwen3`.

```bash
curl http://spark-r0:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-flash-next-4p89bpw",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 512
  }'
```

Notes on request parameters:
- `reasoning_effort: "none"` disables thinking and puts the answer in
  `content` directly. Otherwise reasoning lands in `reasoning_content`.
- Vision: standard OpenAI `image_url` content blocks work (1 image per
  prompt by default, configured `--limit-mm-per-prompt '{"image":1}'`).

## Start / stop / status

Everything runs from the vLLM checkout on the Spark. All launch knobs are
environment variables; the command below is the proven config.

```bash
ssh spark-r0
cd ~/projects/vllm

# START (TP1, the tuned run-4 profile)
rm -f ~/serve_qwen38.log ~/serve_launch.log
nohup env \
  PATH="$HOME/venvs/qwen38/bin:/usr/local/cuda/bin:/usr/bin:/bin" \
  VLLM_PLE_TABLE_MEMORY=disk \
  B12X_POLICY_MODE=auto \
  MODEL_PATH=$HOME/models/Qwen3.8-Flash-Next-NVFP4 \
  LOG_FILE=$HOME/serve_qwen38.log \
  MAX_NUM_SEQS=16 \
  KV_CACHE_MEMORY_BYTES=21474836480 \
  MAX_MODEL_LEN=262144 \
  MAX_NUM_BATCHED_TOKENS=8192 \
  CUDAGRAPH_CAPTURE_SIZES=auto \
  bash serve-qwen38-flash-next-nvfp4-tp1.sh -- \
    --mamba-ssm-cache-dtype bfloat16 \
    --gdn-prefill-backend flashinfer \
    --gdn-decode-kernel cuda \
  > $HOME/serve_launch.log 2>&1 &

# STATUS
curl -s http://127.0.0.1:8000/health -o /dev/null -w '%{http_code}\n'
pgrep -f vllm.entrypoints          # server processes
tail -f ~/serve_qwen38.log         # engine log

# STOP
pkill -f vllm.entrypoints; sleep 5; pkill -9 -f EngineCore
```

Startup takes ~5 minutes (PLE table validation + b12x kernel warmup +
CUDA graph capture). Watch for `Application startup complete` in the log.

## What each knob does

| Knob | Value | Effect |
|---|---|---|
| `VLLM_PLE_TABLE_MEMORY` | `disk` | PLE n-gram table read from SSD via io_uring (frees 26.8 GiB for KV). `ram` = faster steps but 26.8 GiB pinned → KV drops to ~557k tokens. |
| `MAX_NUM_SEQS` | 16 | Concurrent streams. 16 needs full CUDA graphs at every verify width. |
| `KV_CACHE_MEMORY_BYTES` | 21474836480 (20 GiB) | KV pool = 1.39M tokens. Raise only if host has headroom (MemAvailable floor ~6 GiB). |
| `MAX_MODEL_LEN` | 262144 | Native context ceiling (no YaRN). |
| `MAX_NUM_BATCHED_TOKENS` | 8192 | Prefill chunk width. |
| `CUDAGRAPH_CAPTURE_SIZES` | `auto` | FULL graphs sized to seqs × MTP width (128 tokens) — every verify batch gets a graph. |
| `--mamba-ssm-cache-dtype bfloat16` | | Halves SSM state bandwidth; +5–12% decode. |
| `--gdn-prefill-backend flashinfer` | | Required with bf16 SSM (b12x GDN prefill is FP32-only). |
| `--gdn-decode-kernel cuda` | | Fused decode kernel (pairs with flashinfer prefill). |

## Health checks after any restart

```bash
# 1. health endpoint
curl -s http://127.0.0.1:8000/health -o /dev/null -w '%{http_code}\n'
# 2. KV pool line in the log — should read 1,392,826 tokens
grep "KV cache size" ~/serve_qwen38.log | tail -1
# 3. b12x kernels ready (71/71 for gemm; check last "b12x ready" line)
grep "b12x ready" ~/serve_qwen38.log | tail -1
# 4. smoke completion
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-flash-next-4p89bpw","messages":[{"role":"user","content":"17*23=? number only"}],"max_tokens":200,"temperature":0,"reasoning_effort":"none"}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip())'
# expect: 391
```

## Common problems

| Symptom | Cause / fix |
|---|---|
| `Engine core initialization failed` | Read the `EngineCore` block in `~/serve_qwen38.log` (root cause is above the APIServer traceback). |
| `NV_ERR_NO_MEMORY` in dmesg | GPU over-commit. Lower `KV_CACHE_MEMORY_BYTES` or `MAX_NUM_SEQS`. |
| MemAvailable < 6 GiB | Too much pinned/host memory. Check `free -g`; reduce KV or drop PLE to `disk`. |
| Slow C=1 decode vs earlier | Confirm bf16 SSM flags present and PLE is `disk`, not `device`. |
| 64k+ prompts rejected | `MAX_MODEL_LEN` is 262144 but bench prompts+generation must fit; check the error text. |
| Need YaRN 512k | `--rope-scaling` config change required; native ceiling is 262k on this checkpoint (see README "Context" notes). |

## Benchmarking

```bash
ssh spark-r0
cd ~/projects/llm-inference-bench
~/venvs/qwen38/bin/python llm_decode_bench.py \
  --host 127.0.0.1 --port 8000 --model qwen3.8-flash-next-4p89bpw \
  --no-hw-monitor --concurrency 1,2,4,8,16 \
  --contexts 8192,32768,65536,131072 \
  --prefill-contexts 8k,32k,64k,128k \
  --max-tokens 2048 --duration 30 --display-mode plain \
  --output ~/bench_$(date +%m%d).json
```

Coding speed (single-stream, long codegen):
```bash
~/venvs/qwen38/bin/python ~/bench/coding_bench.py   # prints tok/s + TTFT
```

## Key paths on spark-r0

| Path | Contents |
|---|---|
| `~/projects/vllm` | vLLM @ `dev/karmic-kraken` + local patches (branch `frspec-draft-vocab`, revert commit `89d4b75f`) |
| `~/projects/b12x` | b12x source (editable install) |
| `~/models/Qwen3.8-Flash-Next-NVFP4` | Checkpoint (98.6 GiB, 36 shards) |
| `~/venvs/qwen38` | Python env (torch 2.13.0+cu132, flashinfer, b12x) |
| `~/projects/frspec` | Parked FR-Spec work + draft vocab file (51k ids) |
| `~/bench/*.json` | All benchmark raw data |
| `~/serve_qwen38.log` | Engine log (current run) |

## TP2 (when the second Spark arrives)

See `TP2_MIGRATION.md`. Nothing about this install changes; you add the
second node, set `HEAD_IP`/`WORKER_IP`, and run the branch's RDMA launcher
(`--check` first). PLE disk reads and collectives then split across both
nodes — the TP1 PLE-I/O cost (~7% at C=1) mostly disappears.
