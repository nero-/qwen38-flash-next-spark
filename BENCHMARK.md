> **Current TP2 profile: `hc-adaptive`, MTP3**, selected by the user; `hc` is the backup.

Final cable trial (2026-09-24): **two cables selected, adaptive HC/MTP3 unchanged**. Large collective times improved 4–6%; model results were modest and mixed. [Measured comparison and one-cable fallback](tp2/optimization/CABLES.md).
> Adaptive decode showed no repeatable advantage; both profiles span roughly
> 221–232 tok/s at 32K/C8. The HC prefill improvement remains the useful result.
> See [the guide](tp2/README.md) and [controlled results](tp2/optimization/HC-SEPARATION.md).

# Single-Spark LIL benchmarks

Use the unmodified [local-inference-lab/llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench), v0.6.2 at `ccd9ad8ced7e387794391bfb0ac6d99b1f66ba6f`.

The source build updated September 23 uses vLLM `cff8aebfeb73de96c97b4c5a0550c5a49f27451c` and b12x `0332cc5089137753d3af43d1c643516b4350359f`. The b12x loader now uses ordinary CUDA allocations. Its result is approximately unchanged from the earlier InstantTensor run when normalized for MTP acceptance; these measurements do not show a doubling of speed.

| Context | Earlier Instant prefill | Updated b12x prefill | Earlier Instant C1 | Updated b12x C1 | Earlier Instant C8 aggregate | Updated b12x C8 aggregate |
|---|---:|---:|---:|---:|---:|---:|
| 8k | 2,481 | 2,466 | 36.54 | 39.33 | 119.59 | 114.73 |
| 32k | 2,469 | 2,457 | 36.85 | 37.06 | 115.72 | 117.87 |
| 64k | 2,348 | 2,347 | 39.33 | 35.14 | 114.49 | 118.08 |

All rates are tokens/s. C8 aggregate is the sum across eight active requests; divide by eight for mean per-stream throughput. These are single matrix runs, not estimates of run-to-run variance. C1 output rates vary with MTP acceptance: normalized request rounds/s changed by roughly −1% to +1.4% across the six cells. At C8, the harness's normalized rounds/s are aggregate request rounds, not global engine batch steps.

## Method and configuration

The six decode cells use 8,192 / 32,768 / 65,536 nominal context, concurrency 1 and 8, 30-second measurement windows, max 2,048 generated tokens, upstream default sampling, and no reasoning override. Standalone cold prefill uses client prompt-tokens/TTFT at the same nominal lengths. Token targeting is approximate; actual inputs are recorded in each report. Both runs completed all six cells with zero errors, matched effective concurrency, and no underfill, warmup timeout, capacity limitation, or detected exact repetition loop.

Both use the same checkpoint, full vocabulary, MTP 3, disk PLE, 262,144 max context, 20 GiB FP8 KV, max 16 sequences, 8,192-token chunks, BF16 recurrent state, FlashInfer GDN prefill, and CUDA GDN decode. No new quantization or vocabulary restriction was introduced. Host toolkit: CUDA 13.3.1; source PyTorch: 2.13.0+cu132; driver: 580.178.04.

The updated source build passed 252 targeted upstream tests (9 skipped), accepted five images, and retrieved all three passkeys from a 261,888-token prompt. These are functional checks, not a comprehensive quality evaluation or KLD measurement.

Evidence: [source matrix](results/tp1-opt-20260922/kk-b12x-cu133-matrix.json), [exact command](results/tp1-opt-20260922/kk-b12x-cu133-matrix-command.json), [launch settings](results/tp1-opt-20260922/kk-b12x-cu133-launch.json), [matrix summaries](results/tp1-opt-20260922/lil-matrix-summary.json).

## Container comparison: same serving settings

Requested image: `eugr/spark-vllm-b12x:nightly-20260923`, pinned to `sha256:5249a162cd39aa090e803243aef376e1f87f0fd4826f9249ea1ecd0c814b2c7e`.
Actual installed packages: vLLM `0.1.dev21504+g57fdda71b.d20260923`, b12x 1.3.0, PyTorch 2.13.0+cu130, CUTLASS DSL 4.7.0, FlashInfer 0.7.0. Bundled nvcc reports 13.0.88. This compares a complete packaged stack, not an isolated Docker or CUDA change. Host CUDA 13.3 remains installed.

All six image cells passed with zero errors and no underfill, warmup timeout, capacity limitation, or detected exact loop. The packaged stack did not improve the primary matrix: prefill was within approximately 1.3% of source and normalized decode request rounds changed by −4.1% to +0.9%. Raw decode token rates vary with acceptance.

| Context | Source prefill | Image prefill | Source C1 | Image C1 | Source C8 aggregate | Image C8 aggregate |
|---|---:|---:|---:|---:|---:|---:|
| 8k | 2,466 | 2,457 | 39.33 | 34.59 | 114.73 | 117.39 |
| 32k | 2,457 | 2,425 | 37.06 | 35.84 | 117.87 | 116.32 |
| 64k | 2,347 | 2,334 | 35.14 | 37.05 | 118.08 | 124.87 |

Official Coding Peak on the image (Sieve of Eratosthenes, three runs, model-default sampling/thinking, max 2,000 output tokens): **50.69 tok/s mean**, 47.59–53.94 range. This uses completion-token counts and excludes TTFT from generation rate. It is a different workload from the long-context matrix. The developer's reported 60→90 tok/s is not yet reproduced; MTP 3 matches, but his exact prompt/context and remaining settings are unavailable.

Image evidence: [matrix](results/tp1-opt-20260922/eugr-20260923-r2-matrix.json), [launch](results/tp1-opt-20260922/eugr-20260923-r2-launch.json), [Coding Peak](results/tp1-opt-20260922/eugr-20260923-r2-coding.json). The image also accepted five images and correctly counted them.

The updated source build averaged **52.50 tok/s** on the same three-run Coding Peak test (50.10–56.34 range), versus the image's 50.69. These small, stochastic samples overlap; they do not establish a coding speed advantage for the image.

The attempted isolated BF16 b12x decode switch was rejected before loading: this vLLM build requires b12x prefill and decode together, and b12x prefill requires FP32 state. The supported paired b12x/FP32 experiment also completed, increasing recurrent-state precision while retaining the same model, MTP, context and KV byte budget.

| Context | b12x/FP32 prefill | C1 decode | C8 aggregate decode |
|---|---:|---:|---:|
| 8k | 2,323 | 35.64 | 115.70 |
| 32k | 2,316 | 37.51 | 110.42 |
| 64k | 2,210 | 33.59 | 108.90 |

All six cells passed. Its prefill was approximately 5% slower than the matched CUDA/BF16 image profile. Coding Peak averaged **51.42 tok/s** (48.34–53.29), again with no substantial gain. It is not the selected performance profile. [FP32 matrix](results/tp1-opt-20260922/eugr-20260923-r2-b12x-fp32-matrix.json), [FP32 coding](results/tp1-opt-20260922/eugr-20260923-r2-b12x-fp32-coding.json).

Across this entire image/source-recheck session, system used memory peaked at **112.55 GiB**; the 119 GiB guard never triggered. The five-image check passed on the CUDA/BF16 image. No lower precision, reduced vocabulary or smaller context was used.

## Completed TP1 kernel comparison

The same CUDA/BF16 container was tested on 7.0.0-1019 and 6.17.0-1032,
with driver 580.178.04 unchanged. On 6.17, C1 measured 40.26 / 36.55 / 35.99
at 8K / 32K / 64K, C8 aggregate 117.81 / 116.98 / 119.40, and prefill
2479 / 2473 / 2348 tokens/s. Coding Peak averaged 54.75 tokens/s
(range 49.12–60.55). Acceptance-normalized C1 improved about 2–5%; these
single runs do not isolate kernel effects from reboot and sampling variation.
The one-time 6.17 boot has ended; both Sparks currently run 7.0 for TP2.

See [TP2 results and configuration](tp2/README.md) for the current deployment.

## Historical corrections and retention

The former custom coding benchmark counted SSE chunks instead of tokens. Its reported ~16.4 tok/s is invalid and must not be used. MiaAI's short prose, sampling, and thinking settings differ from this LIL matrix; its rates are not a matched comparison. Historical analysis remains in [PERFORMANCE_AUDIT.md](PERFORMANCE_AUDIT.md) and [TP1_OPTIMIZATION.md](TP1_OPTIMIZATION.md).

Old raw benchmark artifacts were deleted at the user's request; retain the latest two completed comparison runs. Earlier rates above are historical summaries. Operational launch commands are in [OPERATIONS.md](OPERATIONS.md).
