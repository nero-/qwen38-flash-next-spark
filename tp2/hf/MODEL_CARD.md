---
library_name: vllm
pipeline_tag: image-text-to-text
base_model: local-inference-lab/Qwen3.8-Flash-Next-NVFP4
base_model_relation: merge
license: other
license_name: qwen-community-1.0
license_link: LICENSE
tags:
- nvfp4
- mxfp8
- quantization-aware-distillation
- b12x
- dgx-spark
---

# Qwen3.8-Flash-Next NVFP4 QAD-5500 Hybrid

A tensor-level merge of two revisions of
[local-inference-lab/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4).
It keeps every **trained** tensor from the newer step-5500 QAD checkpoint and stores
the two **untrained** module groups in the serving formats of the earlier release,
so it runs on the b12x kernel stack at the earlier release's speed.

| Component | Source revision | Stored as |
|---|---|---|
| Routed experts, shared experts, routers, HC mixing, PLE tables/projections/norms, all other trained text tensors | [`qad-step5500-ple1000`](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4/tree/60215d26cf5e42c2db6128774032d57fc62678da) (`60215d26`) | as published: NVFP4 experts, MXFP8 shared experts, NVFP4 PLE |
| Text attention projections incl. QSA indexers (240 modules) | [`main`](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4/tree/7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd) (`7c4f1bc1`) | MXFP8, block 32 (step-5500 ships these in BF16) |
| MTP routed experts | `main` (`7c4f1bc1`) | W4A16 NVFP4 (step-5500 ships MXFP8) |
| MTP non-expert tensors, embeddings, LM head, vision, tokenizer, templates | step-5500 | unchanged (MTP non-expert tensors are byte-identical in both revisions) |

## Why

Both upstream model cards state that attention and the MTP module were **not**
distilled: the earlier release stores attention as MXFP8 post-training quantization,
step-5500 stores the same source attention in BF16, and both quantize the same source
MTP experts (NVFP4 vs MXFP8). On GB10 (SM121) the published step-5500 has two costs:
BF16 attention adds weight reads to every decode step, and b12x has no MXFP8 MoE
kernel, so its MTP drafter needs a slower backend (Triton has no SM121 MXFP8 kernel;
Marlin works). Restoring the earlier formats for exactly these untrained modules
keeps all step-5500 training while serving at full b12x speed.

The MTP head only proposes draft tokens; speculative verification uses the target
model, so the drafter format affects speed, never the target model's outputs.

## Verification

- Attention: every sampled `main` MXFP8 tensor dequantizes to the step-5500 BF16
  tensor with 2.65% ± 0.03% relative RMS error, the expected MXFP8 rounding error
  for identical source weights (an unrelated source would be ~100%).
- MTP: all 29 non-expert MTP tensors are byte-identical between the two revisions.
- Build: 3,312 tensors removed (240 BF16 attention weights, 3,072 MXFP8 MTP expert
  tensors) and 5,088 added (480 MXFP8 attention weight/scale tensors, 4,608 NVFP4 MTP
  expert tensors). The quantization config lists the restored modules as `MXFP8` and
  `W4A16_NVFP4`. `HYBRID.json` records every replaced module. Two machines built
  byte-identical trees independently.

## Measured on 2 × DGX Spark (TP2)

vLLM `dev/karmic-kraken` (`57fdda71b`, image `eugr/spark-vllm-b12x` nightly
2026-09-23) with b12x `8a99d639`, tensor parallel 2 over RoCE, MTP3
(probabilistic draft, block verification), FP8 KV, BF16 recurrent state,
262,144-token context. LIL `llm_decode_bench` v0.6.2, default sampling with
thinking, single samples per cell. Same session, same stack:

| | `main` | This hybrid | Published step-5500 (Marlin MTP) |
|---|---:|---:|---:|
| Prefill 8K / 64K tok/s | 3718 / 3405 | 3671 / 3349 | 3666 / 3359 |
| C1 decode 8K tok/s (steps/s) | 59.4 (26.7) | 65.1 (26.5) | 49.8 (23.0) |
| C8 aggregate 8K tok/s | 231.5 | 231.5 | 219.1 |
| C16 aggregate 8K / 32K tok/s | 342.6 / 332.0 | 344.6 / 328.9 | – |
| MMLU-Pro, 1,000 q, direct answer | 67.2% | 66.9% | 67.0% |

Speed matches `main` within measurement noise (single cells vary ±3–5%; C1 also
moves with draft acceptance, which averaged 2.36 vs 2.34 tokens per step here).
Our evaluations (wikitext/code NLL, GSM8K, MMLU-Pro) could not distinguish the three
checkpoints; the reason to prefer this one is the longer distillation, not a
measured quality gain.

With 24 GiB FP8 KV per rank it held eight concurrent 258,000-token sessions:
95.3% prefix-cache hits on follow-up turns, 74% peak KV use, no preemptions, peak
host memory 102.5 / 99.5 GiB.

## Serving notes

- Requires a runtime with `modelopt_mixed` support for NVFP4 routed experts, MXFP8
  attention/shared experts, W4A16 NVFP4 MTP experts and NVFP4 PLE (for example the
  Local Inference Lab vLLM fork with b12x, `--quantization modelopt_mixed
  --moe-backend b12x --linear-backend b12x`).
- The hybrid has only been run on the stack above; local-inference-lab has not
  reviewed or qualified it. Their step-5500 manifest also notes full-model serving
  was not run upstream.
- Identical requests are not bitwise reproducible on this stack (per-token
  log-probabilities vary by ~0.2 nats between repeats with every checkpoint tested),
  which limits how finely quality differences can be measured.

## License and credit

Weights derive from [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
via local-inference-lab's quantization-aware distillation and remain under the
[Qwen Community License 1.0](LICENSE), included unchanged; its commercial-use
conditions apply to this derivative. All distillation and quantization work is by
local-inference-lab and Qwen; this repository only recombines their published tensors.
