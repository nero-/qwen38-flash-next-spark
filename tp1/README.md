# One DGX Spark (TP1)

Serves the same pinned stack as the [two-Spark deployment](../tp2/README.md) on a single
DGX Spark / ASUS Ascent GX10: image
`eugr/spark-vllm-b12x@sha256:5249a162…2c7e` (vLLM `57fdda71b`, b12x 1.3.0) and the
step-5500 QAD hybrid checkpoint
[JMNI-Labs/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid](https://huggingface.co/JMNI-Labs/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid)
at revision `87c8f2fb`, verified against [`../tp2/manifests/model-5500h.sha256`](../tp2/manifests/model-5500h.sha256).

RESULTS_PLACEHOLDER

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
