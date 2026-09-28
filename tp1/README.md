# One DGX Spark (TP1)

Serves the same pinned stack as the [two-Spark deployment](../tp2/README.md) on a single
DGX Spark / ASUS Ascent GX10: image
`eugr/spark-vllm-b12x@sha256:5249a162…2c7e` (vLLM `57fdda71b`, b12x 1.3.0) and the
step-5500 QAD hybrid checkpoint
[JMNI-Labs/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid](https://huggingface.co/JMNI-Labs/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid)
at revision `87c8f2fb`, verified against [`../tp2/manifests/model-5500h.sha256`](../tp2/manifests/model-5500h.sha256).

## Selected profile: `tp1+cg4+mxhead` — measured 2026-09-28 on gx10-r3

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
serving user, Python 3 with `venv`, `git`, ~110 GiB free disk. From the Mac, set `ssh` in
[node-config.json](node-config.json) to the Spark, then:

```bash
bash tp1/bootstrap.sh     # copy package, pull image, download + verify checkpoint, install harness
bash tp1/preflight.sh     # read-only check
./spark1-ctl.sh start && ./spark1-ctl.sh wait
```

Bootstrap never starts a server and refuses to run while `qwen-tp1` or `qwen-tp2` is
running on the host. The checkpoint is public; set `HF_TOKEN` on the Spark only if you
point the config at a gated snapshot.

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
python3 campaign.py --prefix P tp1:matrix:q tp1+cg4:quick
python3 summarize_arms.py TAG...
```
