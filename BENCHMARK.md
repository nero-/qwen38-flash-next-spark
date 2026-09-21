# Qwen3.8-Flash-Next NVFP4 (QAD) — DGX Spark TP1 Benchmark Report

**Host:** spark-r0 (DGX Spark, GB10, 121 GiB unified LPDDR5X, CUDA 13.0 driver 580.178.04)
**Engine:** vLLM `0.1.dev21460+gaf9e4dca1` (`local-inference-lab/vllm` branch `dev/karmic-kraken`) + b12x 1.3.0 source build (SM121)
**Checkpoint:** `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` — 98.6 GiB, quantization-aware distillation (QAD): NVFP4 routed experts/PLE, MXFP8 attention + shared experts, W4A16 NVFP4 vision FC2/MTP

## Config history

| | Run 1 (initial) | Run 2 (current) |
|---|---|---|
| PLE n-gram table (26.8 GiB) | `device` (GPU-resident) | `disk` (SSD io_uring, `VLLM_PLE_TABLE_MEMORY=disk`) |
| KV pool | 1.5 GiB = 68,457 tok (492 blocks) | **20 GiB = 912,912 tok (6,561 blocks)** |
| `max_num_seqs` | 1 | **8** |
| Max concurrency @65,536 ctx | 1.04x | **13.93x** |
| Engine RSS | 93.7 GiB | 67.2 GiB (−26.5 GiB = PLE moved to SSD) |

Current serve command:
```bash
PATH="$HOME/venvs/qwen38/bin:$PATH" VLLM_PLE_TABLE_MEMORY=disk \
MAX_NUM_SEQS=8 KV_CACHE_MEMORY_BYTES=21474836480 bash serve.sh
```

## Run 2 results (PLE→SSD, KV 20 GiB, seqs 8)

### Prefill (integrated scout)

| ctx | TTFT | prefill tok/s |
|---|---|---|
| 8k | 4.02 s | **2,040** |
| 32k | 15.85 s | **2,029** |

(unchanged vs Run 1 — PLE rows come from SSD only during decode lookup; prefill reads the same checkpoint shards.)

### Sustained decode, aggregate tok/s

| ctx \ conc | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| 8k | **36.0** | 54.5 | 66.1 | **103.7** | 99.7 |
| 32k | **31.8** | 48.2 | 72.5 | 99.0 | **104.3** |

Per-stream at C=8: ~13.9 tok/s (8k), ~13.4 (32k). Compare Run 1: 34.4 @C=1 and *no* working concurrency cells. At C=16 aggregate matches C=8 — the scheduler runs 8 concurrent streams (`max_num_seqs=8`), the extra 8 queue.

MTP-normalized: ~16 steps/s at C=1 (accept ~2.1); at C=8 accept ~2.0 with 50 tok/s aggregate.

### Cells still not measured

- 64k/128k rows: server `max_model_len=65536` refuses 64k-context prompts + 2k generation (HTTP 400). `MAX_MODEL_LEN=131072` would enable them; KV pool (913k tokens) fits 128k × 7 streams.
- C=16 aggregate caps at C=8 level — raise `MAX_NUM_SEQS=16` to push past 100 tok/s (KV budget supports it: 913k ÷ (16×65k) → 0.87x at full context; fine for 8–32k traffic).

### Max coding speed (single-stream, long codegen prompt, thinking off, temp 0)

| run | tokens | decode tok/s |
|---|---|---|
| 1 | 465 | **15.67** |
| 2 | 410 | **15.64** |
| 3 | 422 | **15.61** |

**~15.6 tok/s** — unchanged by the offload (single-stream decode is bandwidth-bound on the same weights; the PLE table's disk rows are only touched during n-gram lookup and the cache absorbs repeat hits). This matches MiaAI's measured single-stream *prose* numbers when normalized for their larger accept length (they hit 48.7 tok/s at accept 3.0/step = ~16 steps/s; we run ~16 steps/s at accept 2.1 → 33–36 tok/s prose).

## Comparison with MiaAI-Lab single-Spark recipe (measured on equivalent hardware)

| metric | MiaAI (2026-09-06 sweep) | ours (Run 2) | delta |
|---|---|---|---|
| decode C=1 prose | 48.7 tok/s (accept 3.0) | 36.0 tok/s (accept 2.1) | steps/s: 16.2 vs 17.1 — parity within noise |
| decode C=8 aggregate | 162.9 tok/s | 103.7 tok/s | −36% — their max_num_seqs=8 with FULL graphs at every verify width (4..32); ours captures fewer widths and schedules 8/16 queued |
| KV pool | ~16.7 GiB (992k FP8 tokens) | 20 GiB (912,912 tokens) | parity (theirs 1,132k at 16 GiB wish — block-size differences) |
| prefill | 2,200 tok/s @8k | 2,040 tok/s @8k | −7% |

Remaining gap at C=8 is graph coverage + scheduler policy, not KV or PLE. Candidate levers: `CUDAGRAPH_CAPTURE_SIZES=auto` equivalent (capture every (1+MTP)×S width), `MAX_NUM_SEQS=16`, and the reduced-vocabulary draft head (FR-Spec) which MiaAI credits +25% single-stream decode — the branch's MTP head here runs the full 248k vocab (VLLM_MTP_NVFP4_LM_HEAD=1) with no vocab-subset support in this fork's Qwen4Exp MTP.

## Run 1 (kept for reference)

- Decode: 34.4 tok/s @C=1; C≥2 cells never ran (max_num_seqs=1 + admission stall) or were KV-skipped (32k+, pool 68k tokens)
- Prefill: 2,114 tok/s @8k
- Coding: 15.86 tok/s

## Notes

- PLE disk mode: b12x `DiskTable` (io_uring, queue depth 64, O_DIRECT) over the 16 checkpoint shards carrying `ple_embedding.ngram_embedding.shard_*`; rows register from file offsets — no repacking, no copy. `prepare_disk` runs outside CUDA graph capture per engine step (batch-bounded compact row cache).
- Engine memory: weights ~71 GiB GPU-side + runtime; PLE no longer in RSS. Host MemAvailable ~15 GiB after load.
- Raw JSON: `~/bench_decode2.json` (run 2), `~/bench_decode.json` (run 1), `~/coding_speed.json` on spark-r0.
