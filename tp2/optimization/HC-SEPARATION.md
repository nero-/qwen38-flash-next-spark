# HC separation and kernel dispatch — 2026-09-24

Packaging note: the controlled follow-up below records an earlier return to the
original HC selection. The final deployment preference is `hc-adaptive`/MTP3,
as selected by the user for this package; the comparison does not establish a
reliable performance winner. `hc` remains the profile backup.

Scope: two bounded experiments, MTP3, unchanged precision, resident PLE,
256K context, MTU9000, same pinned image and checkpoint. MTP5 remains an
optional non-thinking coding-speed profile. No further game-quality evaluation
is required for this speed workload.

## Hypotheses and isolation

1. `hc-prefill`: retain owned-row HC prefill, disable the custom channel-decode
   workspace with `QWEN_HC_CHANNEL_DECODE=0`. Small decode batches use full HC
   projections on each rank, avoiding the two HC channel all-gathers. This is
   **replicated decode**, not an exact restoration of the stock image's decode:
   the stock image also enables HC channel TP by default. Existing `hc` behavior
   remains the default when this new switch is absent.
2. `hc-prefill-dynamic`: same HC isolation plus
   `B12X_MICRO_DYNAMIC_CUTOVER_PAIRS=0`, using the existing dynamic MoE kernel
   instead of micro for tiny batches. Larger batches already use dynamic.
   Arithmetic formats/scales and model capacity are unchanged.

The earlier trace attributes 66.75 ms of decode kernel time to dynamic MoE,
58.54 ms to the two leading dense-GEMM specializations and 25.82 ms to custom
collectives. These are overlapping summed GPU durations, not wall-time shares.
B12x autotuning is already enabled. Dense split-K turbo and packed-B expand-ahead
are already enabled. The alternative dense atom layout does not change the
small M16/M32 tiles, so it is not a compelling first decode experiment.
MX-FP6 fused quantization targets a different format. The NVFP4 split-decode
source explicitly warns about unresolved residual-scale numerics; it is excluded
under the quality constraint.

## Validation

Use unmodified LIL v0.6.2. The isolation trial uses the complete 8K/32K/64K,
C1/C8 matrix with standalone prefill. The kernel trial first uses the shorter
8K/32K C1 screen. Compare acceptance and step rate as well as tokens/sec.
Functional checks are regression signals, not proof of global quality equivalence.
GPU ownership tests cover both channel and replicated decode branches.
Results and final selection are appended after completion.

## Initial results

| Profile | PP 8K / 32K / 64K | C1 8K / 32K / 64K | C8 8K / 32K / 64K |
|---|---|---|---|
| Previous HC | 3737 / 3525 / 3406 | 63.47 / 59.94 / 54.76 | 220.18 / 222.60 / 218.09 |
| Replicated HC decode | 3775 / 3201 / 3424 | 59.17 / 57.87 / 53.38 | 231.63 / 231.33 / 223.61 |

All six replicated-decode cells completed without errors. The same five of seven
functional checks passed as the previous HC profile. Both HC dispatch branches
passed 128 GPU projection/ownership tests before the kernel trial.
C8 engine-step rates improved from 95.72/92.91/92.94 to 96.84/98.16/96.10;
C1 fell from 26.43/26.19/25.96 to 25.55/25.55/25.34. This supports a batch-size
tradeoff; not all the token-rate movement is explained by acceptance.

The all-dynamic MoE override measured C1 56.13 / 57.89 at 8K/32K, versus
59.17 / 57.87 for normal dispatch on the same replicated-HC layout. It is
rejected, despite 6/7 functional checks passing. No kernel override is promoted.

The final bounded follow-up is `hc-adaptive`: original channel HC decode through
four tokens (one MTP3 verification batch), replicated decode above four tokens,
and unchanged owned-row prefill. This uses
`QWEN_HC_CHANNEL_DECODE_MAX_TOKENS=4`; the default remains 128 for reproducibility
of the original `hc` profile. No user batch is truncated or resized.

## Final selection: hc-adaptive

PP 8K/32K/64K: **3739 / 3488 / 3412 tok/s**.
C1: **60.06 / 59.18 / 61.78**. C8 aggregate: **224.90 / 230.86 / 219.30**.
All six cells completed without errors. C1 engine steps/s: 26.42/26.27/25.69;
C8 request-step equivalents/s: 96.08/95.59/92.41. C1 64K's token-rate increase
mostly reflects acceptance (2.407 versus 2.109), not faster engine execution.
C8 token rates are +2.1%/+3.7%/+0.6% versus previous HC; these single samples
support only a modest gain, with the clearest step-rate improvement at 32K.
Prefill is effectively unchanged from previous HC. Original baseline still has
better C8 at some contexts, with materially slower prefill.

256 GPU tests passed for both HC decode modes and cutoffs 4/128, TP2/TP4,
multiple batch sizes, pending residuals and owned rows. The functional suite
passed 6/7; all previously passing checks remain passing, JSON formatting also
passed, and the pre-existing box-swap logic failure remains. These checks do
not establish comprehensive quality equivalence. No precision/quantization or
model-capacity reduction was introduced.

Peak used RAM during this experiment sequence: r0 **98.82 GiB**, r1
**94.38 GiB**, below 119 GiB. Temporary memory guards were stopped.
The model remains running on hc-adaptive. Mac `profile balanced` selects it;
`profile coding` retains the prior hc-k20-mtp5 non-thinking speed option.
Rejected trial profiles remain named and reproducible; no trial runs in the
background. Per-profile caches are a few hundred MiB, not duplicate checkpoints.

Evidence: `results/tp2/hc-adaptive-final-*`, `hc-adaptive-unit.log`,
`hc-prefill-isolated-*`, `hc-prefill-dynamic-screen-*`, and
`hc-isolation-memory-r{0,1}.jsonl`. The generator reproduces modified source,
tests and checksum manifest exactly.

## Repeatability audit (supersedes the initial gain claim)

Two additional unmodified-LIL C8/32K runs used the already warm, unchanged
`hc-adaptive` service. All had eight active requests, zero errors and no timeout.

| Run | Aggregate tok/s | Request-step equivalents/s | Acceptance length |
|---|---:|---:|---:|
| hc-adaptive-final-matrix | 230.86 | 95.59 | 2.415 |
| hc-adaptive-repeat32-c8-1 | 224.39 | 94.23 | 2.381 |
| hc-adaptive-repeat32-c8-2 | 221.31 | 95.05 | 2.328 |

Mean 225.52 tok/s; range 221.31–230.86.
The previous HC result was 222.60 tok/s. The observed spread overlaps that result;
a dependable decode improvement is **not established**. Selection remains
provisional. The next useful validation is interleaved, matched HC/adaptive runs
at 32K/C8, rather than another unrelated flag sweep. Prefill and full-capacity
operation remain validated, but this is not proof of optimal performance.

Kernel source audit: the BF16 small-N GEMV API only routes output widths <=1024,
explicitly excluding the vocabulary head. `VLLM_LM_HEAD_A16` is read inside the
runtime-quantized head branch, so it does not optimize the unchanged BF16 target
head when `VLLM_MXFP8_LM_HEAD=0`. It still applies to the NVFP4 draft head. A future
BF16 target-head kernel experiment must preserve full vocabulary and compare
actual operators at the TP2 shard shapes; toggling this flag is not that test.
No additional runtime changes were made during this audit.

## Controlled HC follow-up: restore original HC default

Adaptive repeats were followed by two original-HC runs and one return-leg
adaptive run, all 32K/C8 with the same official LIL command and no concurrent
external requests at trial start. All completed without errors or underfill.

| Run | Aggregate tok/s | Request-step equivalents/s | Acceptance length |
|---|---:|---:|---:|
| hc-ab-hc-1 | 223.74 | 93.81 | 2.385 |
| hc-ab-hc-2 | 231.85 | 96.19 | 2.410 |
| hc-ab-hc-adaptive-1 | 231.35 | 97.23 | 2.379 |

The previous-HC token range (223.74–231.85) overlaps the adaptive range
(221.31–231.35 across repeat/control runs). Engine-step rates overlap as well.
The return-leg adaptive step rate is higher, but this small unrandomized sample
does not separate dispatch effects from warmup/order/run variation. There is
no robust basis for replacing the original HC default.

Historical decision at the time of this experiment: `balanced` returned to `hc`
(MTP3, owned-row prefill, original channel HC decode). That preference was later
superseded by the user's final package selection of `hc-adaptive`. The selected
precision, PLE placement, full vocabulary, context and image are unchanged.
MTP5 remains optional via `coding`. Measurements remain inconclusive about a
repeatable decode speed winner.
