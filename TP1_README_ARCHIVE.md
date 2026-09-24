> **TP2 deployment:** Both Sparks use resident PLE and the `hc-adaptive` MTP3 default (`hc` is the backup).

Final cable trial (2026-09-24): **two cables selected, adaptive HC/MTP3 unchanged**. Large collective times improved 4–6%; model results were modest and mixed. [Measured comparison and one-cable fallback](tp2/optimization/CABLES.md).
> Use [the TP2 operations guide](tp2/README.md) and `~/Agent/Builds/spark-ctl.sh` on the Mac.
> The TP1 instructions below are historical; do not start them alongside TP2.

# Qwen3.8-Flash-Next NVFP4 (QAD) on DGX Spark — TP1 deployment

Serves `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` on a single DGX Spark
(GB10, 121 GiB unified memory) with vLLM from the
[`local-inference-lab/vllm`](https://github.com/local-inference-lab/vllm/tree/dev/karmic-kraken)
`dev/karmic-kraken` branch and the
[`b12x`](https://github.com/local-inference-lab/b12x) SM121 kernel library.

Current measurements are in [BENCHMARK.md](BENCHMARK.md); the image and pending
kernel comparison are documented there and in [KERNEL_AB.md](KERNEL_AB.md).
Historical settings and experiments are in [RECIPE.md](RECIPE.md),
[PERFORMANCE_AUDIT.md](PERFORMANCE_AUDIT.md), and
[TP1_OPTIMIZATION.md](TP1_OPTIMIZATION.md). The serving wrapper defaults to
256k context, disk PLE, the updated b12x ordinary CUDA loader, MTP 3 and the validated
BF16/GDN settings, using the CUDA 13.3 toolkit. Disk PLE is retained for this
performance-testing round. `PLE_MODE=resident` selects a smaller KV/sequence
profile; its current 4k chunk setting still needs full validation below the
119 GiB ceiling. The earlier resident 2k profile passed a 256k request;
resident 8k exceeded that ceiling.

## Why this checkpoint

Quantization-aware distillation (QAD), not post-training quantization: the
NVFP4/MXFP8 weights were trained against the BF16 teacher (2,500 trunk +
1,500 joint-refinement updates). GPQA Diamond 89.9 beats NVFP4 PTQ while using
9% fewer tokens on average (15% median, 26% P99).

| Component | Format |
| --- | --- |
| Routed experts (512/layer, 10 active) | NVFP4 (group 16) |
| N-gram PLE tables | NVFP4 |
| Shared experts | MXFP8 |
| Attention + QSA indexers | MXFP8 |
| Vision FC2, MTP experts | W4A16 NVFP4 |
| Embeddings, norms, recurrent | BF16 |
| Target LM head at runtime | MXFP8 weights, BF16 activations |
| MTP LM head at runtime | NVFP4 weights, BF16 activations |

## Layout

- venv: `~/venvs/qwen38` (torch 2.13.0+cu132, vLLM editable, b12x editable)
- vLLM: `~/projects/vllm` @ `dev/karmic-kraken` (`cff8aebfe`, clean source; old PLE patch no longer applied)
- b12x: `~/projects/b12x` from `master` (`0332cc5`)
- Weight loader and compute backend: updated b12x. InstantTensor 0.2.0 remains installed as an alternative (`LOAD_FORMAT=instanttensor`).
- CUDA toolkit: 13.3.1 (`/usr/local/cuda-13.3`); driver 580.178.04.
  PyTorch's packaged runtime remains CUDA 13.2.
- Checkpoint: `~/models/Qwen3.8-Flash-Next-NVFP4` (98.6 GiB, 36 shards)

## Install

The Spark is already installed. Use the Mac launch command in
[OPERATIONS.md](OPERATIONS.md); do not rerun the bootstrap to restart serving.
`install.sh` is for a fresh environment and follows moving upstream branches,
so it is not a pinned reproduction of the current deployment.

```bash
bash install.sh
```

Requires: Ubuntu 24.04 aarch64, CUDA 13.3 toolkit (`/usr/local/cuda-13.3`),
`python3.12-dev` (installed via sudo), ~150 GiB disk.

## Serve

```bash
bash serve.sh          # TP=1 on this Spark, port 8000
bash serve.sh --check  # print the command, do not run
```

Downloads nothing at launch; uses `~/models/Qwen3.8-Flash-Next-NVFP4`.

## TP2 (two-Spark cluster)

When the DAC arrives, the second Spark becomes a TP=2 build using the same venv and
checkpoint — nothing about this install is single-Spark-only:

```bash
# on the head node, from the vllm checkout:
B12X_ROOT=~/projects/b12x \
VLLM_ROOT=~/projects/vllm \
MODEL_PATH=~/models/Qwen3.8-Flash-Next-NVFP4 \
HEAD_IP=<head-ip> WORKER_IP=<worker-ip> \
bash scripts/serve-qwen38-flash-next-nvfp4-tp2-rdma.sh --check
```

- Transport: RoCE over the direct ConnectX-7 link; b12x one-shot all-reduce
  (`rocenante`), NCCL fallback via `ALLREDUCE=nccl`.
- Interfaces: `rocep1s0f0`, `rocep1s0f1`, `roceP2p1s0f0`, `roceP2p1s0f1`
  (verify with `ibdev2netdev`).
- `--sync-code` mirrors vllm/ + b12x/ to the worker; `--sync-model` rsyncs the
  ~99 GiB checkpoint.
- TP=2 roughly doubles KV headroom vs TP=1 (~2.7x concurrency at 262k context).
- Falls back to vLLM's b12x PCIe all-reduce backend
  (`VLLM_ENABLE_PCIE_ALLREDUCE=1 VLLM_PCIE_ALLREDUCE_BACKEND=b12x`) if not
  using the RDMA launcher.

## Verification

```bash
bash smoke.sh   # /health + one completion
```
