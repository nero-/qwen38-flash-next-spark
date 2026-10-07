# One DGX Spark (TP1)

This model is based on [Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD) by Local Inference Lab, Inc., a non-profit organization, available at <https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD>. Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD is licensed under the Local Inference Lab License, Version 1.0.

Serves Qwen3.8-Flash-Next on a single DGX Spark / ASUS Ascent GX10 from the
NVFP4-CSF container
[local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD)
at revision `ac1c8d8e`, verified file by file against its `SHA256SUMS`, whose own hash is
pinned in [node-config.json](node-config.json). The container is a lossless re-encoding of
the step-5500 QAD hybrid the [two-Spark deployment](../tp2/README.md) serves: all 40 source
shards it records match our hybrid manifest byte for byte, and only the routed experts' E4M3
block scales are compressed (7.55 → 3.97 GB). Do not reupload or mirror it; link to the
repository above (LIL License 1.0).

The NVFP4-CSF reader is newer than any published image, so the serving image `kk1007` is
built locally with eugr's B12X build pinned to vLLM `e07345714` (karmic-kraken), b12x 1.5.0
`2cc7f66a` and FlashInfer `f8d3729e` (the commit in eugr's `nightly-20261006`); see
[Serving image](#serving-image).

## Selected profile: `tp1+cg4+mxhead+kk1007+csf+kv24` — measured 2026-10-07 on gx10-r3

The 09-28 profile on the `kk1007` image with the CSF checkpoint, and the memory CSF frees
put back into KV: **24 GiB FP8 KV = 1,509,026 tokens (5.76 × 262,144)**, up from 1,257,472
(4.80 ×), at a lower peak than today's deployment. Kernel 6.17.0-1032-nvidia, driver
580.178.04. Same harness and cells as below; all arms ran in one session. Engine steps/s
isolates speed from MTP acceptance.

| | Previous (`5249a162`, hybrid, 20 GiB) | `kk1007`, hybrid, 20 GiB | `kk1007`, CSF, 20 GiB | **Selected: CSF, 24 GiB** |
|---|---:|---:|---:|---:|
| Prefill 8K / 32K / 64K tok/s | 2410 / 2392 / 2281 | 2444 / 2432 / 2316 | 2385 / 2378 / 2264 | 2393 / 2372 / 2265 |
| C1 steps/s 8K / 32K / 64K | 17.68 / 17.53 / 17.39 | 18.15 / 17.98 / 17.92 | 18.11 / 17.82 / 17.84 | 17.68 / 17.64 / 17.37 |
| C1 tok/s 8K / 32K / 64K | 40.7 / 44.0 / 44.2 | 42.1 / 42.4 / 38.1 | 41.5 / 44.6 / 42.2 | 40.9 / 44.9 / 39.0 |
| C8 steps/s 8K / 32K / 64K | 62.5 / 58.7 / 58.4 | 62.5 / 62.4 / 60.9 | 61.8 / 59.6 / 57.9 | 60.7 / 61.0 / 60.1 |
| C8 aggregate tok/s | 145.2 / 140.8 / 149.5 | 147.7 / 149.2 / 147.1 | 154.8 / 146.6 / 149.5 | 144.2 / 149.1 / 144.6 |
| C3 / C5 / C16 (8K, 32K) tok/s | 86.8 / 114.9 / 209.5, 212.5 | 85.8 / 111.9 / 215.1, 216.0 | 87.7 / 114.2 / 212.6, 219.0 | – |
| Wiki NLL · code NLL · GSM8K · MMLU-Pro | 1.395 · 0.150 · 97.6% · – | 1.396 · 0.150 · 97.6% · 65.2% | 1.397 · 0.150 · 96.8% · 66.2% | (same weights) |
| Peak used RAM | 114.21 GiB | 112.61 GiB | 109.55 GiB | **113.33 GiB** |

- **CSF is lossless and costs ~2% of prefill.** Against the hybrid on the same image: wiki
  NLL +0.0006 ± 0.0020, code +0.0008 ± 0.0013, GSM8K two questions differ, MMLU-Pro 32 vs 22
  discordant in CSF's favor (McNemar p = 0.22). Prefill is 2.2–2.4% slower at every context,
  consistent with expanding scales into the shared 150 MiB scratch before large prefill
  batches; decode steps are flat to noisy (±5% single cells).
- **The new image** is equal or slightly faster than `5249a162` on the hybrid (C1 steps
  +2–3%) with the same quality, and peaks 1.6 GiB lower.
- **Memory:** CSF + new image peak 4.66 GiB below today's deployment at 20 GiB KV; 24 GiB
  KV brings the peak back to 113.33 GiB, under the previous 114.21. The previous
  deployment's baseline MMLU-Pro run was stopped before completing; its paired NLL/GSM8K
  against the selected weights is +0.0015 ± 0.0018 / +0.0007 ± 0.0012 / 3 vs 1 discordant.
- A 5 × 258K concurrent-session capacity check at 24 GiB was not run.

Receipts: [`results/tp1-csf-20261007/`](../results/tp1-csf-20261007/summary.json)
(harness reports, the image's build metadata, generated summary); full evidence on the Spark
under `~/builds/qwen-tp1/evidence/` with tags `c2base-*`, `c2-*` and `final-kv24`.

## Previous selection: `tp1+cg4+mxhead` — measured 2026-09-28 on gx10-r3

Base `tp1` plus exact MTP3 CUDA-graph sizes (`cg4`) and MXFP8 target LM-head weights
(`mxhead`). 20 GiB FP8 KV = **1,257,472 tokens (4.8 × 262,144)**; peak system RAM under
the full benchmark **113.4 GiB** of 121 GiB. Kernel 6.17.0-1032-nvidia, driver 580.178.04.

Unmodified LIL v0.6.2 (`ccd9ad8c`), same cells as the TP2 campaign: 30 s windows, max
2,048 output tokens, model-default sampling and thinking, cold standalone prefill.
C8/C16 are aggregate throughput. Single cells vary ±3–5%, and C1 tok/s moves with MTP
acceptance, so compare engine steps/s as well.

| | Previous TP1 best (Sep 23, 6.17) | Base `tp1` | Selected |
|---|---:|---:|---:|
| Prefill 8K / 32K / 64K tok/s | 2479 / 2473 / 2348 | 2454 / 2466 / 2350 | 2460 / 2398 / 2312 |
| C1 tok/s 8K / 32K / 64K | 40.3 / 36.6 / 36.0 | 40.5 / 40.7 / 40.3 | 39.2 / 41.7 / 43.6 |
| C1 steps/s 8K / 32K / 64K | – | 17.52 / 17.50 / 17.29 | 17.76 / 17.49 / 17.48 |
| C8 aggregate 8K / 32K / 64K | 117.8 / 117.0 / 119.4 | 142.4 / 141.6 / 143.5 | 145.5 / 143.3 / 143.0 |
| C8 steps/s 8K / 32K / 64K | – | 58.72 / 59.20 / 58.88 | 60.46 / 61.00 / 56.94 |
| C3 / C5 aggregate at 8K | – | 81.6 / 111.0 | 89.4 / 115.7 |
| C16 aggregate 8K / 32K | – | 213.5 / 206.5 | 219.7 / 206.5 |
| Wiki NLL · GSM8K · MMLU-Pro | – | 1.397 · 97.6% · 66.2% | 1.395 · 97.2% · 67.3% |
| Peak used RAM | 112.6 GiB | 112.8 GiB | 113.4 GiB |

The main gain over the previous single-Spark deployment (+21% C8 at unchanged prefill and
RAM) comes from the stack shared with TP2: the step-5500 hybrid checkpoint, probabilistic
drafting with block verification, and BF16 recurrent state. The two modifiers add a small
further gain, clearest at C3–C8. Credit has not been split among the base-stack changes.

Quality uses TP2's frozen inputs (`quality_eval.py`, corpus `2e42176f`, MMLU subset
`169665a7`). Selected vs base, paired: wiki NLL −0.0020 ± 0.0022, code NLL −0.0003 ±
0.0015, GSM8K one question differs, MMLU-Pro 39 vs 28 discordant (McNemar p = 0.22). All
are inside TP2's repeat-run noise floor (wiki Δ 0.0047 on identical baselines, 71 MMLU flips).
The base-stack quality matches TP2's selected profile (1.397 · 96.8% · 66.9%).

### Arm screening

| Arm | Change vs base | Result | Decision |
|---|---|---|---|
| `+cg4` (full matrix) | exact verify-size graphs | steps/s C1 −0.6%, C5 +1.8%, C8 +1.3%; +0.9 GiB RAM | keep |
| `+mxhead` (quick + quality) | MXFP8 head weights | steps/s C1 +2.6%, C8 +5.2%; NLL/GSM8K/MMLU within noise (MMLU p = 0.38) | keep |
| `+greedyspec` | earlier TP1 sampler | not run; superseded | – |

At TP1 the whole 248,320 × 2,560 target head is read every decode step (TP2 reads half per
rank), so halving its bytes is worth more here than it was on TP2, where the BF16 head was
kept. Resident PLE was not re-tested: on one Spark it leaves room for only ~4 GiB of KV and
one sequence, and the earlier TP1 A/Bs found no decode gain.

Harness reports and a generated summary are in [`results/tp1-single-20260928/`](../results/tp1-single-20260928/summary.json). Full receipts (commands, before/after metrics, RAM streams, inspect output)
are on the Spark under `~/builds/qwen-tp1/evidence/` with tags `c1-tp1`, `c1-tp1_cg4`,
`c1-tp1_mxhead`, `c1-tp1_cg4_mxhead`.

## Install

Prerequisites on the Spark: NVIDIA driver + Container Toolkit, Docker usable by the
serving user, Python 3 with `venv`, `git`, ~130 GiB free disk (96 GiB checkpoint, 25 GB
image). From the Mac, set `ssh` in [node-config.json](node-config.json) to the Spark, then:

```bash
bash tp1/bootstrap.sh     # copy package, check/pull image, download + verify checkpoint, install harness
bash tp1/preflight.sh     # read-only check of the selected profile's image and checkpoint
./spark1-ctl.sh start && ./spark1-ctl.sh wait
```

Bootstrap fetches what the default profile needs: the CSF container (verified against
`SHA256SUMS`, then `serve-csf/` built by [prepare_csf.py](prepare_csf.py)), or the hybrid
for non-`csf` profiles (`--all-models` fetches both), and pulls the profile's image by
digest. Bootstrap never starts a server and
refuses to run while `qwen-tp1` or `qwen-tp2` is running on the host. Both checkpoints are
public; set `HF_TOKEN` on the Spark only for a gated snapshot.

## Serving image

`kk1007` is published as
[`ghcr.io/jmni-labs/qwen38-spark-vllm:kk1007-csf`](https://github.com/orgs/JMNI-labs/packages/container/package/qwen38-spark-vllm)
and pinned in `node-config.json` by digest (`sha256:66d59ede…`, image ID `sha256:a03faca5…`),
so bootstrap pulls it like any other image, on any Spark. The image contains no model
weights. To rebuild it, stop `qwen-tp1` and run on the Spark:

```bash
bash ~/builds/qwen-tp1/images/build-kk-csf.sh   # ~95 min; vLLM and FlashInfer compile
```

The script checks out [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) at
`73f01ce`, pins InstantTensor to 0.2.0 (0.2.1, released 2026-10-06, breaks eugr's
InstantTensor patch step), pins vLLM and b12x to the commits above instead of the branch
heads, runs eugr's `--exp-b12x --rebuild-vllm` build with FlashInfer `f8d3729e`, and checks
the image for the pinned commits and the CSF reader. A rebuild gets a new image ID: push it
(`docker tag` + `docker push` to `ghcr.io/jmni-labs/qwen38-spark-vllm`, after `docker login
ghcr.io` with a `write:packages` token) and pin the new digest in
`experimental_images.kk1007`. The deployed image was built in two steps (the same build,
then the runner stage again after the InstantTensor pin) from identical inputs; the
one-pass script has not been run end to end. Once an eugr nightly carries the CSF reader,
pin it by digest instead.

To go back to the hybrid, select a profile without `csf` (bootstrap or `download_model.py
hybrid` fetches it), or restore the original shards from the container with LIL's
`trellis_quant.lossless_scale_checkpoint restore`. Profiles without an image modifier use
`eugr/spark-vllm-b12x@sha256:5249a162…`, which Docker pulls on demand.

## Operate

```bash
./spark1-ctl.sh start | stop | status | wait | logs
./spark1-ctl.sh tunnel                 # http://localhost:8000/v1, model qwen3.8-flash-next-4p89bpw
./spark1-ctl.sh profile default        # the selected profile in node-config.json
./spark1-ctl.sh profile tp1+mxhead     # any tp1[+modifier...] composition
```

A profile switch refuses active requests, saves the old container's inspect/log receipts
under `evidence/`, restarts, waits for readiness, and restores the previous profile if the
new one fails to start. Each profile compiles into its own `cache-<profile>/` directory,
so the first start of a new profile takes longer than later ones.

## Profiles

The base `tp1` profile: TP1, disk PLE, KV from `kv_cache_gib`, 16 sequences, 8,192
scheduler tokens, 262,144 context, FP8 KV, BF16 recurrent state (FlashInfer GDN prefill,
CUDA GDN decode), b12x linear/MoE/loader, BF16 target-head weights, MTP3 with
probabilistic drafting and block verification, full vocabulary, unlimited images per
prompt. Modifiers each change one axis:

| Modifier | Change |
|---|---|
| `cg4` | capture every MTP verify size exactly (multiples of depth+1) |
| `mxhead` | MXFP8 target LM-head weights |
| `greedyspec` | vLLM default draft/rejection sampling (the earlier TP1 setting) |
| `fp32ssm` | checkpoint-native FP32 recurrent state |
| `resident` | PLE table resident in memory; requires an explicit `kvN` |
| `csf` | serve the NVFP4-CSF container (`--quantization nvfp4_csf --load-format nvfp4_csf`); needs an image with the reader |
| `kk1007` | serving image from `experimental_images` (the locally built CSF image) |
| `kvN` | N GiB FP8 KV instead of `kv_cache_gib` |
| `mtpN` | N speculative tokens |
| `seqsN` | N max concurrent sequences |

The TP2 HC overrides (`hc`, `hc-adaptive`) are deliberately absent: channel-parallel HC
decode requires TP > 1 and sharded HC prefill requires world size 2 or 4, so at TP1 they
are inert.

## Measuring

On the Spark, from `~/builds/qwen-tp1`:

```bash
python3 bench.py TAG --cases matrix odd c16     # unmodified LIL v0.6.2 + RAM peak
python3 quality_eval.py run TAG                 # NLL, GSM8K, MMLU-Pro (same frozen inputs as TP2)
python3 quality_eval.py compare TAG_A TAG_B
python3 capacity_check.py TAG --sessions 4      # N concurrent 258K sessions; then --followup
python3 tokenizer_check.py TAG                  # tokenizer time vs cold TTFT on ~250K tokens; token-ID and detokenizer hashes
python3 campaign.py --prefix P tp1:matrix:q tp1+cg4:quick
python3 summarize_arms.py TAG...
```
