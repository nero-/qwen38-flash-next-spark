# Qwen3.8-Flash-Next NVFP4 (QAD) — DGX Spark TP1 Benchmark Report

**Host:** spark-r0 (DGX Spark, GB10, 121 GiB unified LPDDR5X, CUDA 13.0 driver 580.178.04)
**Engine:** vLLM `0.1.dev21460+gaf9e4dca1` (`local-inference-lab/vllm` branch `dev/karmic-kraken`) + b12x 1.3.0 source build (SM121)
**Checkpoint:** `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` — 98.6 GiB, quantization-aware distillation (QAD): NVFP4 routed experts/PLE, MXFP8 attention + shared experts, W4A16 NVFP4 vision FC2/MTP

## Final config (run 4)

```bash
PATH="$HOME/venvs/qwen38/bin:$PATH" \
VLLM_PLE_TABLE_MEMORY=disk \
MAX_NUM_SEQS=16 \
KV_CACHE_MEMORY_BYTES=21474836480 \
MAX_MODEL_LEN=262144 \
MAX_NUM_BATCHED_TOKENS=8192 \
CUDAGRAPH_CAPTURE_SIZES=auto \
bash serve.sh -- --mamba-ssm-cache-dtype bfloat16 \
                 --gdn-prefill-backend flashinfer \
                 --gdn-decode-kernel cuda
```

| | Run 1 | Run 2 | Run 3 | Run 4 (final) |
|---|---|---|---|---|
| PLE table (26.8 GiB) | GPU | SSD io_uring | SSD | SSD |
| KV pool | 1.5 GiB / 68k tok | 20 GiB / 913k | 20 GiB / 1.29M | 20 GiB / **1.39M** |
| `max_num_seqs` | 1 | 8 | 16 | 16 |
| `max_model_len` | 65,536 | 65,536 | **262,144** | 262,144 |
| `max_num_batched_tokens` | 2,048 | 2,048 | **8,192** | 8,192 |
| FULL CUDA graphs | 5 | 9 | **9 (widths 4–128)** | 9 |
| SSM state dtype | float32 | float32 | float32 | **bfloat16** |
| GDN prefill | b12x | b12x | b12x | **flashinfer** (bf16 state) |
| GDN decode | b12x | b12x | b12x | **cuda (fused)** |

Notes:
- `mamba_ssm_cache_dtype=bfloat16` requires `--gdn-prefill-backend flashinfer` — the b12x `gdn_prefill` kernel hard-requires FP32 state. FlashInfer's GDN prefill qualifies on SM12x + head_dim 128 + CUDA 13. The b12x decode kernel also hard-requires FP32, hence the fused CUDA decode kernel (`--gdn-decode-kernel cuda`; b12x and cuda decode both accept bf16, but b12x was verified only with the b12x prefill pairing).
- Capture sizes: `seqs=16 × MTP query_len=4` → ceiling `min(16×4×2, 512)=128`; FULL graphs cover 4–128 tokens, so every verify batch (16 seqs × 4 tok = 64) has an exact graph.
- KV 1,392,826 tokens (bf16 SSM state shrinks the mamba page → 5.31x concurrency at 262,144 ctx).

## Run 4 results (final config)

### Prefill

| ctx | TTFT | prefill tok/s |
|---|---|---|
| 8k | 3.32 s | **2,473** |
| 32k | 13.04 s | **2,468** |
| 64k | 27.9 s | **2,351** |
| 128k | 60.9 s | **2,150** |

### Sustained decode, aggregate tok/s

| ctx \ conc | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| 8k | 33.3 | 55.9 | 71.6 | 116.7 | **162.2** |
| 32k | 33.2 | 53.9 | 69.3 | 105.1 | **161.2** |
| 64k | 33.0 | 54.9 | 75.4 | 107.4 | **163.7** |
| 128k | 33.1 | 54.7 | 79.0 | 108.9 | **156.4** |

All 20 cells measured — no skips, no errors. Per-stream at C=16: ~11 tok/s. TTFT at C=16 ~530 ms.

MTP-normalized steps/s: C=1 ≈ 15.4, C=8 ≈ 55, **C=16 ≈ 82** (accept ~2.0).

### Max coding speed (single-stream, long codegen prompt, thinking off, temp 0)

| run | tokens | decode tok/s |
|---|---|---|
| 1 | 452 | 16.48 |
| 2 | 466 | 16.44 |
| 3 | 522 | 16.41 |
| 4 | 460 | 16.44 |

**~16.4 tok/s** (+5% over run 2's 15.6 — the bf16 SSM state halved the recurrent-state bandwidth).

## Scorecard vs MiaAI-Lab single-Spark recipe (identical hardware, 2026-09-06 sweep)

| metric | MiaAI | ours (run 4) | delta |
|---|---|---|---|
| decode C=1 prose | 48.7 tok/s (accept 3.0 → 16.2 steps/s) | 33.3 tok/s (accept 2.2 → 15.1 steps/s) | **engine steps/s parity (−7%)**; tok/s gap is acceptance, see below |
| decode C=8 aggregate | 162.9 | 116.7 | −28% |
| decode C=16 aggregate | (not measured; C=8 profile) | **162.2** | matches their C=8 number |
| prefill @8k | 2,200 | **2,473** | **+12%** |
| prefill @32k | 2,304 | **2,468** | **+7%** |
| prefill @128k | 2,146 | 2,150 | parity |
| KV pool | ~16.7 GiB / 992,584 tok | 20 GiB / **1,392,826 tok** | +40% |
| context | 262k native / 512k YaRN | 262k native (262k = checkpoint native ceiling) | — |

## Acceptance-length gap — root cause

MiaAI's 3.0 tok/step comes from **reduced-vocabulary drafting (FR-Spec)**: a 47k-token draft-head subset concentrates draft probability mass (their measured +25% single-stream decode, and their published acceptance 0.80/0.59/0.41). Our per-position rates (0.73/0.55/0.43 recent) are ~8% lower at each position on the full 248,320-token draft head.

Verified experiments:
- **NVFP4 vs BF16 draft head** (`VLLM_MTP_NVFP4_LM_HEAD=1` vs `0`): accept 2.18 vs 2.27 median — **no difference** (A/B: C=1 33.3 vs 32.5, C=8 116.7 vs 113.4). Keep NVFP4 (less memory).
- The fork's Qwen4Exp MTP (`vllm/models/qwen4_exp/nvidia/mtp.py`) has **no vocab-subset support**; porting FR-Spec would require upstream work (draft-head row gathering + rejection-sampling mask).

## Run history

| run | config | C=1 | C=8 | C=16 | prefill 8k | KV tok |
|---|---|---|---|---|---|---|
| 1 | GPU PLE, seqs 1, KV 1.5G | 34.4 | — | — | 2,114 | 68k |
| 2 | SSD PLE, seqs 8, KV 20G | 36.0 | 103.7 | 99.7 | 2,040 | 913k |
| 3 | + seqs 16, 256k ctx, batch 8k | 34.7 | 100.3 | 150.4 | 2,324 | 1.29M |
| 4 | + bf16 SSM, FI prefill, fused decode | 33.3 | 116.7 | **162.2** | **2,473** | **1.39M** |

## Remaining levers (not reachable on this fork today)

1. **FR-Spec vocab-subset drafting** — the acceptance lever (theirs 3.0 vs ours 2.2); needs porting the draft-vocab row-slicing into the fork's MTP speculator.
2. **MAX_NUM_SEQS beyond 16** — C=16 ≈ C=8+40%; further scaling likely KV/scheduler-bound at full 262k context (5.3x concurrency ceiling), fine for ≤32k traffic.

Raw JSON: `~/bench_decode4.json` (final), `~/bench_decode3.json` (run 3), `~/bench_decode2.json` (run 2), `~/bench_decode.json` (run 1), `~/coding_speed.json` on spark-r0.
