# Qwen3.8-Flash-Next NVFP4 (QAD) — DGX Spark TP1 Benchmark Report

**Host:** spark-r0 (DGX Spark, GB10, 121 GiB unified LPDDR5X, CUDA 13.0 driver 580.178.04)
**Date:** 2026-09-21
**Engine:** vLLM `0.1.dev21460+gaf9e4dca1` (`local-inference-lab/vllm` branch `dev/karmic-kraken`) + b12x 1.3.0 source build (SM121)
**Checkpoint:** `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` — 98.6 GiB, quantization-aware distillation (QAD): NVFP4 routed experts/PLE, MXFP8 attention + shared experts, W4A16 NVFP4 vision FC2/MTP
**Serving config:** TP=1, `--quantization modelopt_mixed`, `--load-format b12x`, b12x linear/MoE/QSA/GDN backends, MTP 3 (`num_speculative_tokens=3`), FP8 KV, `max_model_len=65536`, `max_num_seqs=1`, `max_num_batched_tokens=2048`, `--kv-cache-memory-bytes 1.5 GiB` (branch default)
**Measured KV pool:** 68,457 tokens (492 blocks × 3008-token block), max concurrency at 65,536 ctx: 1.04x

## Results

### Prefill (integrated scout, client-side prompt_tokens/TTFT)

| ctx | tokens | TTFT | prefill tok/s |
|---|---|---|---|
| 8k | 8,197 | 3.88 s | **2,114** |
| 32k | 32,159 | 15.35 s | **2,096** |

### Sustained decode, aggregate tok/s (5 concurrency × 4 context matrix)

| ctx \ conc | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| 8k | **34.4** | 33.9 | 33.2 | 36.5 | 33.8 |
| 32k | **31.1** | ∅ skip | ∅ skip | ∅ skip | ∅ skip |
| 64k | ERROR | ERROR | ERROR | ERROR | ERROR |
| 128k | ERROR | ERROR | ERROR | ERROR | ∅ skip |

- 8k row: aggregate ≈ per-stream at every concurrency — the scheduler ran 1 stream at a time (max_num_seqs=1 on this config), so concurrency does not multiply throughput. TTFT for queued streams ≈ 30 s (measured wait behind the running stream).
- 32k C≥2 and 128k cells: skipped by KV-budget check (68k-token pool vs 32k/131k per request).
- 64k/128k ERROR cells: bench warmup sent 64k-context requests at a server whose `max_model_len=65536` (prompt + generation exceeded it → HTTP 400). Not an engine fault; a serving-profile mismatch with the requested matrix.

### MTP-normalized decode steps/s (accept length)

| ctx \ conc | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| 8k | 16.0 (2.15) | 16.1 (2.11) | 15.9 (2.09) | 16.1 (2.27) | 15.9 (2.13) |

`steps/s = tok/s ÷ accept_len` — engine forward passes per second, MTP-acceptance-independent. Accept length ≈ 2.1 tokens/step at MTP 3.

### Max coding speed (single-stream, code-generation prompt, thinking off, temp 0)

| run | tokens | decode tok/s |
|---|---|---|
| 1 | 478 | **15.86** |
| 2 | 471 | **15.84** |
| 3 | 494 | **15.86** |

**Max coding speed ≈ 15.9 tok/s** sustained on a long production-quality Python code-generation task (6,000+ chars out). The 34 tok/s prose-decode cells benefit from short 2k-token generations; long-form code generation settles at ~15.9 tok/s.

## Known issues during measurement

1. **Scheduler admission stall under concurrency** — with `max_num_seqs=1` + MTP + the branch's mamba `align` boundary-checkpoint admission path, queued requests at C≥2/32k never got admitted (warmup timeout, 0/2 running). The bench marked those cells as skipped. The 8k C≥2 rows ran (via prefix-cache hits) but the engine ran one stream at a time.
2. **KV pool is small on TP1** — 68,457 tokens total (1.5 GiB KV budget from the branch default `KV_CACHE_MEMORY_BYTES=1610612736`). This is why 32k-concurrency and 128k cells did not fit. Raising `KV_CACHE_MEMORY_BYTES` (e.g. 8–16 GiB) on a TP1 box would unlock 32k/64k concurrency cells; 128k requires >16 GiB.
3. **64k/128k ERROR cells** are `max_model_len=65536` refusals, not engine failures. Set `MAX_MODEL_LEN=131072` to measure them.

## Notes

- `num_requests_waiting` climbed to 16 during the bench (all queued streams), and the engine logger showed "Running: 0–1, Waiting: 15–16" — consistent with max_num_seqs=1.
- Prefill ~2.1k tok/s at 8k–32k is in line with single-stream NVFP4 MoE on GB10; the 128k row was not measurable due to the context-length refusal.
- The server has been restarted with defaults; to reproduce with a larger KV pool: `KV_CACHE_MEMORY_BYTES=17179869184 MAX_MODEL_LEN=131072 bash serve.sh`.
