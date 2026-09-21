# Qwen3.8-Flash-Next NVFP4 (QAD) on DGX Spark — TP1 deployment

Serves `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` on a single DGX Spark
(GB10, 121 GiB unified memory) with vLLM from the
[`local-inference-lab/vllm`](https://github.com/local-inference-lab/vllm/tree/dev/karmic-kraken)
`dev/karmic-kraken` branch and the
[`b12x`](https://github.com/local-inference-lab/b12x) SM121 kernel library.

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
| Embeddings, LM head, norms, recurrent | BF16 |

## Layout

- venv: `~/venvs/qwen38` (torch 2.13.0+cu132, vLLM editable, b12x editable)
- vLLM: `~/projects/vllm` @ `dev/karmic-kraken` (`af9e4dc`)
- b12x: `~/projects/b12x` @ `master` (`0f3a8cb`)
- Checkpoint: `~/models/Qwen3.8-Flash-Next-NVFP4` (98.6 GiB, 36 shards)

## Install

```bash
bash install.sh
```

Requires: Ubuntu 24.04 aarch64, CUDA 13.x (`/usr/local/cuda`), Docker (not
needed for serving), `python3.12-dev` (installed via sudo), ~150 GiB disk.

## Serve

```bash
bash serve.sh          # TP=1 on this Spark, port 8000
bash serve.sh --check  # print the command, do not run
```

Downloads nothing at launch; uses `~/models/Qwen3.8-Flash-Next-NVFP4`.

## TP2 (two-Spark cluster)

Tomorrow's second Spark becomes a TP=2 build using the same venv and
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
