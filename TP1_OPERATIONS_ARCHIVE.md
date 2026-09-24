> **TP2 deployment:** Both Sparks use resident PLE and the `hc-adaptive` MTP3 default (`hc` is the backup).

Final cable trial (2026-09-24): **two cables selected, adaptive HC/MTP3 unchanged**. Large collective times improved 4–6%; model results were modest and mixed. [Measured comparison and one-cable fallback](tp2/optimization/CABLES.md).
> Use [the TP2 operations guide](tp2/README.md) and `~/Agent/Builds/spark-ctl.sh` on the Mac.
> The TP1 instructions below are historical; do not start them alongside TP2.

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
- Vision: standard OpenAI `image_url` content blocks work. The wrapper passes
  `--limit-mm-per-prompt '{}'`, replacing the upstream one-image cap with
  vLLM's default of 999 per modality. Five images were tested successfully;
  context and memory still limit practical request size.

## Active kernel test and container

The current kernel test uses `qwen38-eugr-20260923-r2-kernel617`, the pinned
September 23 image with CUDA-GDN/BF16 state and MTP 3. Kernel 6.17 was selected
for one boot; the permanent default remains 7.0. See [KERNEL_AB.md](KERNEL_AB.md)
for the comparison and boot behavior.

From the Mac, when no other server is using port 8000:

```bash
ssh spark-r0 'docker start qwen38-eugr-20260923-r2-kernel617'
ssh spark-r0 'docker logs --tail 50 -f qwen38-eugr-20260923-r2-kernel617'
ssh spark-r0 'curl -fsS http://127.0.0.1:8000/health'
```

For a local Mac endpoint, keep this SSH tunnel open and use
`http://localhost:8000/v1` with model `qwen3.8-flash-next-4p89bpw`:

```bash
ssh -N -L 8000:127.0.0.1:8000 spark-r0
```

## Pinned container experiment

The September 23 image is pinned by digest in `diagnostics/container_trial.py`.
Its benchmark results and active-backend decision are recorded in BENCHMARK.md.
The container uses its own CUDA 13.0 toolchain; the source installation below
continues to use the host CUDA 13.3 toolkit.

To create a fresh matched container trial from the Mac after stopping an idle
server (choose a new tag each time):

```bash
ssh spark-r0 '~/venvs/qwen38/bin/python ~/projects/qwen38-flash-next-spark/diagnostics/container_trial.py --tag my-image-test'
ssh spark-r0 'docker logs -f qwen38-my-image-test'
```

Use `--gdn-decode-kernel b12x` for paired b12x prefill/decode with FP32
state; this image rejects b12x decode with FlashInfer prefill. The original
matched image matrix used `cuda` with BF16 state. MTP 3, full vocabulary,
context 262144, 20 GiB KV and disk PLE are fixed by this launcher. It runs as user 1000,
mounts the checkpoint read-only, and writes compiler caches only under
`~/.cache/qwen38-eugr-20260923`. Docker's seccomp filter is disabled for the
loader's io_uring calls. No host source tree or CUDA installation is mounted.

To stop either serving backend after active requests finish:

```bash
ssh spark-r0 '~/venvs/qwen38/bin/python ~/projects/qwen38-flash-next-spark/diagnostics/tp1_session.py stop'
```

To restart an existing stopped container, use `docker start <container-name>`;
check that port 8000 is free first. Logs remain available through `docker logs`.
Do not start the source wrapper while a container is serving on that port.

## Source build: start / stop / status

From the Mac, after the existing server has stopped, the installed wrapper
starts the current KK+b12x/CUDA 13.3 profile in the background:

```bash
ssh spark-r0 'nohup env LOG_FILE="$HOME/serve_qwen38.log" bash "$HOME/projects/qwen38-flash-next-spark/serve.sh" > "$HOME/serve_launch.log" 2>&1 < /dev/null &'
ssh spark-r0 'tail -f ~/serve_launch.log'
ssh spark-r0 'curl -s -o /dev/null -w "HTTP %{http_code}\n" http://127.0.0.1:8000/health'
```

The process survives SSH disconnects. HTTP 200 indicates readiness. The
benchmark trial uses its own logs under `~/bench/tp1-opt-20260922`; the log
paths above apply to subsequent starts with this command.

The source profile runs from the vLLM checkout on the Spark. All launch knobs are
environment variables; the command below is the proven config.

```bash
ssh spark-r0
cd ~/projects/vllm

# START (TP1, disk PLE, updated b12x loader; CUDA 13.3 toolkit)
rm -f ~/serve_qwen38.log ~/serve_launch.log
nohup env \
  PATH="$HOME/venvs/qwen38/bin:/usr/local/cuda-13.3/bin:/usr/bin:/bin" \
  CUDA_HOME=/usr/local/cuda-13.3 \
  TRITON_PTXAS_PATH=/usr/local/cuda-13.3/bin/ptxas \
  VLLM_PLE_TABLE_MEMORY=disk \
  B12X_POLICY_MODE=auto \
  OMP_NUM_THREADS=16 NUM_SPECULATIVE_TOKENS=3 \
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
    --load-format b12x \
    --limit-mm-per-prompt '{}' \
  > $HOME/serve_launch.log 2>&1 &

# STATUS
curl -s http://127.0.0.1:8000/health -o /dev/null -w '%{http_code}\n'
pgrep -f vllm.entrypoints          # server processes
tail -f ~/serve_qwen38.log         # engine log

# STOP: inspect the process command, then terminate that API PID gracefully.
# Run after active requests have finished.
ps -eo pid,args | grep '[v]llm.entrypoints.cli.main'
kill -TERM <API_PID>
```

Startup takes ~5 minutes (PLE table validation + b12x kernel warmup +
CUDA graph capture). Watch for `Application startup complete` in the log.

## What each knob does

| Knob | Value | Effect |
|---|---|---|
| `VLLM_PLE_TABLE_MEMORY` | disk | Read PLE rows from the existing checkpoint. Temporary choice during performance testing. |
| `MAX_NUM_SEQS` | 16 | Scheduling capacity; matched matrix benchmarks use one or eight active requests. |
| `KV_CACHE_MEMORY_BYTES` | 21474836480 (20 GiB) | KV pool = 1,257,472 tokens. Monitor total system usage against 119 GiB, including temporary buffers. |
| `MAX_MODEL_LEN` | 262144 | Native context ceiling (no YaRN). |
| `MAX_NUM_BATCHED_TOKENS` | 8192 | Prefill chunk width. |
| `CUDAGRAPH_CAPTURE_SIZES` | `auto` | Capture sizes follow the configured sequence capacity and MTP width. |
| `--load-format b12x` | | Updated loader uses ordinary CUDA allocations and bounded asynchronous I/O. `LOAD_FORMAT=instanttensor bash serve.sh` selects the alternative loader. |
| `CUDA_HOME` | `/usr/local/cuda-13.3` | Explicit toolkit for compilation. PyTorch still packages CUDA 13.2. |
| `--mamba-ssm-cache-dtype bfloat16` | | Halves SSM state bandwidth; +5–12% decode. |
| `--gdn-prefill-backend flashinfer` | | Required with bf16 SSM (b12x GDN prefill is FP32-only). |
| `--gdn-decode-kernel cuda` | | Fused decode kernel (pairs with flashinfer prefill). |

## Health checks after any restart

```bash
# 1. health endpoint
curl -s http://127.0.0.1:8000/health -o /dev/null -w '%{http_code}\n'
# 2. KV pool line in the log — should read 1,257,472 tokens
grep "KV cache size" ~/serve_qwen38.log | tail -1
# 3. b12x kernels ready (counts depend on build; check last "b12x ready" line)
grep "b12x ready" ~/serve_qwen38.log | tail -1
# 4. smoke completion
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-flash-next-4p89bpw","messages":[{"role":"user","content":"17*23=? number only"}],"max_tokens":200,"temperature":0,"reasoning_effort":"none"}' \
  | ~/venvs/qwen38/bin/python -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip())'
# expect: 391
```

## Common problems

| Symptom | Cause / fix |
|---|---|
| `Engine core initialization failed` | Read the `EngineCore` block in `~/serve_qwen38.log` (root cause is above the APIServer traceback). |
| `NV_ERR_NO_MEMORY` in dmesg | GPU over-commit. Lower `KV_CACHE_MEMORY_BYTES` or `MAX_NUM_SEQS`. |
| MemAvailable < 6 GiB | Too much pinned/host memory. Check `free -g`; reduce KV or drop PLE to `disk`. |
| Slow C=1 decode vs earlier | Compare the same prompt, sampling, context and MTP acceptance; confirm BF16 SSM flags and memory headroom. PLE placement alone does not explain throughput. |
| 64k+ prompts rejected | `MAX_MODEL_LEN` is 262144 but bench prompts+generation must fit; check the error text. |
| Need YaRN 512k | `--rope-scaling` config change required; native ceiling is 262k on this checkpoint (see README "Context" notes). |

## Benchmarking

```bash
ssh spark-r0
cd ~/projects/llm-inference-bench
~/venvs/qwen38/bin/python llm_decode_bench.py \
  --host 127.0.0.1 --port 8000 --model qwen3.8-flash-next-4p89bpw \
  --no-hw-monitor --concurrency 1,8 \
  --contexts 8192,32768,65536 \
  --standalone-prefill --prefill-contexts 8k,32k,64k \
  --max-tokens 2048 --duration 30 --display-mode plain \
  --output ~/bench_$(date +%m%d).json
```

Coding Peak (official `local-inference-lab/llm-inference-bench`):
```bash
# Run the updated wrapper from this repository on the Spark.
bash coding_bench.sh
# Optional explicitly greedy variant; report separately from model defaults:
CODING_TEMPERATURE=0 bash coding_bench.sh
```

The wrapper uses the upstream built-in Sieve-of-Eratosthenes prompt and
`--coding-peak`, recording results under `coding_peak` in the output JSON.
It runs a short C=1 decode cell first. Coding Peak leaves thinking at the
server default; `--reasoning-effort` does not control this mode in v0.6.2.
Do not use `~/bench/coding_bench.py`: it counts SSE chunks as tokens.
For cross-recipe comparisons, match sampling, thinking, prompt and context;
the historical main matrix inherited temperature 1.0 and thinking enabled.
See [PERFORMANCE_AUDIT.md](PERFORMANCE_AUDIT.md).

## Key paths on spark-r0

| Path | Contents |
|---|---|
| `~/projects/vllm` | vLLM from `dev/karmic-kraken`, commit `cff8aebfe`, branch `codex/kk-loader-cu133-20260923` |
| `~/projects/b12x` | b12x source at `0332cc5` (editable install) |
| `~/models/Qwen3.8-Flash-Next-NVFP4` | Checkpoint (98.6 GiB, 36 shards) |
| `~/venvs/qwen38` | Python env (torch 2.13.0+cu132, flashinfer, b12x) |
| `~/projects/frspec` | Parked FR-Spec work + draft vocab file (51k ids) |
| `~/bench/tp1-opt-20260922` | Retained benchmark evidence |
| `~/serve_qwen38.log` | Source wrapper log; trials use their own logs |

## TP2 (when the second Spark arrives)

See `TP2_MIGRATION.md`. Nothing about this install changes; you add the
second node, set `HEAD_IP`/`WORKER_IP`, and run the branch's RDMA launcher
(`--check` first). Measure compute, PLE reads and collective overhead after
bring-up; the earlier claim that a 7% PLE cost would disappear was unproven.
