# TP2 Migration Guide — Adding a Second DGX Spark

The TP1 install on spark-r0 is fully reusable for a two-Spark TP=2 cluster.
Nothing about the venv, vLLM build, or checkpoint is single-Spark-specific.

## What you need on day 2

1. **Second Spark** reachable over SSH from spark-r0 (and ideally from your
   workstation) with the same OS level (Ubuntu 24.04 aarch64).
2. **Direct ConnectX-7 link** between the two Sparks (the RDMA path the
   branch's launcher uses). Verify on both nodes:
   ```bash
   ibdev2netdev          # expect rocep1s0f0, rocep1s0f1, roceP2p1s0f0, roceP2p1s0f1
   ibstat                # LinkUp on the direct link
   ```
3. **Same runtime image on both nodes** — the launcher runs everything in
   Docker (`IMAGE_NAME`, default `vllm-node-eugr-20260712:latest`, an amd64
   image name in the script but built for aarch64 in practice; the script
   checks `docker image inspect` locally and on the worker and fails fast if
   missing). If you do not have that image, build one on the Spark from the
   vllm checkout or ask for the branch's image build instructions.

## Paths the launcher expects (all overridable)

| Env var | Default (script author's box) | Use on your cluster |
|---|---|---|
| `VLLM_ROOT` | `~/projects/vllm` (auto) | same |
| `B12X_ROOT` | `/home/luke/projects/b12x` | `~/projects/b12x` |
| `SPARK_ROOT` | `/home/luke/projects/spark-vllm-docker` | wherever you keep `launch-cluster.sh` |
| `CLUSTER_LAUNCHER` | `$SPARK_ROOT/launch-cluster.sh` | required; must be executable |
| `MODEL_PATH` | `/data/models/.../qwen3.8-flash-next-180b-nvfp4-ple-mxfp8-attn-shared_vv1` | `~/models/Qwen3.8-Flash-Next-NVFP4` |
| `HEAD_IP` | `192.168.42.223` | spark-r0's IP on the cluster link |
| `WORKER_IP` | `192.168.42.110` | new Spark's IP |
| `ETH_IF` / `IB_IF` | `enP7s7` / 4 RoCE ifaces | verify with `ibdev2netdev` |
| `CONTAINER_MEMORY_GB` | 108 | keep (host has 121 GiB) |

## Bring-up sequence

```bash
# 1. On spark-r0: clone + build the same stack on the worker, then sync code
#    and checkpoint from spark-r0 (the launcher can do both):
cd ~/projects/vllm
B12X_ROOT=$HOME/projects/b12x \
VLLM_ROOT=$HOME/projects/vllm \
SPARK_ROOT=<path-to-launch-cluster-dir> \
MODEL_PATH=$HOME/models/Qwen3.8-Flash-Next-NVFP4 \
HEAD_IP=<spark-r0-ip> WORKER_IP=<spark1-ip> \
bash scripts/serve-qwen38-flash-next-nvfp4-tp2-rdma.sh --check

# 2. Sync code (vllm + b12x trees) and the 98.6 GiB checkpoint to the worker:
... same env ... bash scripts/serve-qwen38-flash-next-nvfp4-tp2-rdma.sh --sync-code
... same env ... bash scripts/serve-qwen38-flash-next-nvfp4-tp2-rdma.sh --sync-model   # ~99 GiB over the direct link

# 3. Launch (head runs in foreground; use --detach for background):
... same env ... bash scripts/serve-qwen38-flash-next-nvfp4-tp2-rdma.sh
```

`--check` validates topology (SSH reachability, image presence, path
whitespace rules) without launching.

## What changes at TP=2

- `--tensor-parallel-size 2`, `--load-format fastsafetensors` (the b12x
  managed-memory loader is TP1-specific; TP=2 uses shared reads into device
  memory per rank), `--gpu-memory-utilization 0.90`.
- Collectives route through b12x `rocenante` one-shot RoCE all-reduce
  (`ALLREDUCE=rocenante`, default). `ALLREDUCE=nccl` is the fallback.
- KV budget: `KV_CACHE_MEMORY_BYTES=1610612736` (1.5 GiB) default in the
  script; at TP=2 the pool is per-rank, so total KV roughly doubles —
  raise `KV_CACHE_MEMORY_BYTES` if you want the 32k/64k concurrency cells
  that were skipped in the TP1 benchmark.
- vLLM/b12x source must be **identical on both ranks** (the script compares a
  runtime digest and refuses to launch on drift; `--sync-code` fixes it).

## Fallback: non-RDMA TP=2

If you cluster over plain Ethernet/PCIe instead of the ConnectX-7 link, use
vLLM's b12x PCIe all-reduce backend directly (no cluster launcher):

```bash
VLLM_ENABLE_PCIE_ALLREDUCE=1 VLLM_PCIE_ALLREDUCE_BACKEND=b12x \
  vllm serve ~/models/Qwen3.8-Flash-Next-NVFP4 \
  --tensor-parallel-size 2 --quantization modelopt_mixed \
  --gdn-decode-kernel b12x --linear-backend b12x --moe-backend b12x ...
```

(Ray or torchrun multi-node setup required for cross-node TP; the RDMA
launcher exists because it packages that plumbing.)
