# Full-vocabulary BF16 target-head probe — 2026-09-24

The serving profile remains unchanged (`hc-adaptive`, MTP3). This experiment
runs separate synthetic tensors in a short-lived process on idle rank0; it does
not replace model weights or the live head implementation.

## Verified implementation and shapes

The checkpoint has hidden_size=2560, vocab_size=248320 and untied embeddings.
TP2 therefore gives each head shard N=124160, K=2560: 606.25 MiB of BF16 weights.
The image's UnquantizedEmbeddingMethod.apply calls dispatch_unquantized_gemm()
with its default backend; on this CUDA path it resolves to
`default_unquantized_gemm`, which calls `torch.nn.functional.linear`.
The probe uses the equivalent BF16 matrix multiplication with a preallocated
output for graph replay. This isolates the multiply, excluding logits collectives,
sampling, allocator/host effects and all other transformer work.

## Bounded kernel experiment

Compare Torch to a Triton full-BF16 GEMV, FP32 products/reductions, no vocabulary
restriction, with N tiles 4 and 8. Five timing samples each replay eight calls
in a CUDA graph. Seed42 synthetic Gaussian weights and activations; fixed 128
vocabulary rows additionally compared to FP64. This is a screening experiment,
not comprehensive model-quality validation or proof of bit identity.

| Rows | Torch ms | Triton N4 ms | N8 ms | N4 speedup |
|---|---:|---:|---:|---:|
| 1 | 3.421 | 2.448 | 2.457 | 1.40x |
| 4 | 2.478 | 2.426 | 2.468 | 1.02x |
| 32 | 2.557 | not tested | not tested | — |

Both candidates agree on argmax for all tested rows. RMS differences versus
Torch: 9.55e-6 at M1, 1.18e-4 at M4. Maximum differences: .001953 and .015625.
On sampled FP64 reference rows, candidate and Torch RMS errors match at .001843
and .001582 respectively. Different FP32 reduction order can still change BF16
rounding; this is not a lossless-drop-in guarantee. Real checkpoint activations,
extreme-value tests and broader distribution comparisons would be required
before any deployment consideration.

## Decision

Do not deploy. MTP3 normally verifies up to four target rows together. The
four-row improvement is only ~0.052 ms for this operator, compared with roughly
38 ms per observed C1 engine step. Even if that saving transfers one-for-one,
it is only about 0.14% of step time. The M1 improvement is real in this isolated
sample, but is not evidence of a comparable MTP3 decode gain; the draft head
uses NVFP4 and is a different operator. Both rank timings and end-to-end behavior
would require validation before attributing model throughput gains.

The probe narrows the search: prioritize the repeated MoE/HC/dense projections
inside transformer layers and controlled HC A/B validation before investing in
this BF16 head replacement. Keep the prototype as a reproducible experiment,
not a serving dependency. No new package or runtime setting was installed.

Evidence: `results/tp2/bf16-head-probe.json`, `bf16-head-probe.log`.
Source: `tp2/optimization/bench_bf16_head.py` (Torch 2.13.0+cu130, NVIDIA GB10).
