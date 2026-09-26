# Two direct-connected DGX Sparks

## Install or reproduce on another pair

This package targets two DGX Sparks running a mutually compatible NVIDIA driver/Docker runtime, with the NVIDIA Container Toolkit, Docker access as the serving user, Python 3, `ssh`, and `rsync`. The host must expose the selected RoCE HCAs and network interfaces through `/sys`, and the serving user must be able to run Docker commands. These are prerequisites; the bootstrap does not install system packages, alter SSH, or provision the management network.

On the Mac that controls the cluster, clone this repository and edit [cluster-config.json](cluster-config.json). The example contains the current cluster's example management SSH destinations, direct-link IPs, interface/HCA names, two-cable map, pinned image and model revision. Set both `ssh` values to hostnames reachable from the Mac. Set rank 1's `peer_ssh` to the address/alias rank 0 can reach over its management network, and arrange key-based SSH from rank 0 to rank 1. Both ranks should use the same Unix account and relative `remote_root`; peer commands resolve files under the remote account's `$HOME`.

Provision the direct-link IPs on the data NICs yourself before bootstrap; `ranks[].ip` must equal the address assigned to that rank's `socket_interface`, and rank 0 must be able to reach rank 1 there. Confirm the link is up and ping works both ways. Bootstrap does not add addresses, change routes, or configure NetworkManager/netplan. For a new physical cabling, map each selected HCA to the host NIC that exposes it. Keep both rails on each rank on the same RoCE v2 GID index; the runtime discovers an IPv4-mapped RoCE v2 GID from sysfs and rejects ambiguous or mismatched mappings. Set `default_cables` to `1` or `2` only after describing the actual wiring in both `ranks[].cables` maps. The bundled two-cable selection is the tested example topology. One-cable mode selects both HCAs exposed by that physical port.

Before bootstrap, ensure rank 0 can authenticate to rank 1 without an interactive password prompt. For gated or access-controlled model snapshots, make `HF_TOKEN` available to noninteractive commands on each Spark; the bootstrap passes it only to the one-off downloader. Do not put tokens in this repository. Then run from the repository root on the Mac:

```bash
bash install.sh
```

The script checks Docker reachability on both hosts and refuses to run while either `qwen-tp2` container is active. It copies the TP2 runtime files without deleting remote model, cache, receipt, or selection data, pulls the pinned OCI image, independently downloads the exact model commit on each rank, and verifies every non-cache model file against SHA-256 hashes from rank 0. Each rank then downloads the pinned step-5500 trunk, builds the default hybrid checkpoint with `optimization/build_hybrid.py` (which first verifies that both revisions share the same source attention weights), and checks both trees against the manifests in `manifests/`. Expect about 200 GiB of additional disk per rank; hybrid shards that do not change are hardlinked. It initializes profile/cable selections only when they do not already exist. Pass `--reset-selection` only when deliberately replacing those selections with the values in the config. The bootstrap does not change host networking or start serving.

After it completes, run the read-only, hardware-aware preflight on the Mac:

```bash
bash tp2/preflight.sh
```

Preflight validates both ranks' selected links, discovered GIDs, source override hashes, model files, and pinned image availability. A successful preflight is not a clean-install quality or performance qualification.

## Day-to-day operation

The copied [spark-ctl.sh](../spark-ctl.sh) controller on the Mac runs:

```bash
./spark-ctl.sh start
./spark-ctl.sh wait
./spark-ctl.sh status
./spark-ctl.sh profile balanced   # hc-adaptive+cg4+m5500h, selected default
./spark-ctl.sh profile fp32state  # balanced with FP32 recurrent state
./spark-ctl.sh profile previous   # September 24 default: hc-adaptive, main checkpoint
./spark-ctl.sh profile original   # original HC + MTP3
./spark-ctl.sh profile coding     # optional HC/top-20/MTP5 profile
./spark-ctl.sh cables 1           # validate and switch both ranks together
./spark-ctl.sh cables 2
./spark-ctl.sh tunnel             # keep the SSH tunnel open
./spark-ctl.sh stop
```

Profile and cable changes preflight both ranks and reject active requests. They retain stopped container inspection/log receipts, restart the pair together, and wait for API readiness; a failed restart attempts to restore the previous selection. If the request metrics endpoint cannot establish that a running server is idle, the change is refused. `bash tp2/preflight.sh` checks the selected pair without starting a model.

At start, the launcher sets MTU 9000 on selected DAC interfaces only, when needed. It uses a short-lived root Docker helper with `NET_ADMIN`, read-only mounts of the host `ip` binary and runtime libraries, and the same pinned image. Docker permissions must allow this helper. The change is runtime-only; persistent host network files and management interfaces are not edited. Do not change cable mode on only one rank.

The image, model and custom source override hashes are pinned. The overrides preserve their upstream Apache-2.0 headers; the image's and model repository's own license/usage terms remain applicable. See [provenance notes](PROVENANCE.md). Deployment evidence below belongs to the current two-Spark cluster and should not be read as proof of behavior on another cluster.

Both ranks are serving the selected `hc-adaptive+cg4+m5500h` profile (MTP3) with
24 GiB KV per rank. Its full LIL matrix, C16 cells and quality suite completed
without errors, and eight concurrent 258,000-token sessions passed with 0
preemptions (see the September 26 section). 256 HC projection/dispatch tests
passed earlier; five-image and 261,888-token passkey checks passed on the HC port.

Both nodes use `~/builds/qwen-tp2` for launcher, checkpoint, compiler cache and
receipts. Rank 0 retains a compatibility symlink at the old model location.
The existing TP1 containers stay stopped during TP2 operation.

Image: `eugr/spark-vllm-b12x@sha256:5249a162cd39aa090e803243aef376e1f87f0fd4826f9249ea1ecd0c814b2c7e`.
vLLM `57fdda71b`, b12x `8a99d639`.

The baseline profile keeps the original checkpoint and full vocabulary, resident
PLE, BF16 target-head weights and recurrent state, FP8 KV (`kv_cache_gib`, now 24 GiB
per rank for every profile; 20 GiB before September 26),
MTP3 probabilistic drafting with block verification, CUDA GDN decode and
FlashInfer prefill, b12x linear/MoE/loader, context 262144, 16 sequences,
8192 scheduler tokens, and graph-capture cap 64. It does not include the TP4
custom HC patch or top-20 draft filtering. Target MXFP8 and w13 layer-max
scaling experiments are not enabled.

The balanced profile (`hc-adaptive`) uses owned-row HC prefill and adaptive
HC decode with MTP3. The user selected it as the default; original HC (`hc`) is
available as `profile original`. The 64K/C1 acceptance benefit remains unconfirmed,
and repeated 32K/C8 tests showed overlapping performance. This selection is a
preference, not a claim of a proven universal speedup. See
[HC results](optimization/HC-SEPARATION.md).
The experimental coding profile (`hc-k20-mtp5`) adds correctly cached top-20
draft proposals and MTP5, with graph cap 96. It measured 99.18 tok/s on one
8192-token thinking-disabled capped Tetris run versus baseline 89.58, but regressed general
decode in some contexts. The cap truncated the game; this is not completed-task
throughput. With thinking enabled, matched 4096-token reasoning windows measured
69.99 tok/s for HC/MTP3 versus 66.62 for HC/top20/MTP5; keep MTP3. MXFP8 target-head + w13 scaling was tested separately at MTP5 and
is not selected. See [experiment details](optimization/PLAN.md).

Direct-link addresses are 10.100.168.2 / 10.100.169.2 on r0 and
10.100.168.1 / 10.100.169.1 on r1. Bootstrap uses the first direct interface;
RoCEnante/NCCL select both active RDMA devices explicitly for each host.
Both hosts currently run kernel 7.0.0-1019-nvidia, NVIDIA driver 580.178.04.

Mac controller: `~/Agent/Builds/spark-ctl.sh` controls both ranks.

```bash
./spark-ctl.sh start
./spark-ctl.sh wait
./spark-ctl.sh status
./spark-ctl.sh stop
./spark-ctl.sh profile balanced  # Adaptive HC + exact graphs, step-5500 hybrid
./spark-ctl.sh profile fp32state # Balanced with FP32 recurrent state
./spark-ctl.sh profile previous  # September 24 default (main checkpoint)
./spark-ctl.sh profile original  # Original HC backup, MTP3
./spark-ctl.sh profile coding    # Experimental HC/top20/MTP5
./spark-ctl.sh profile baseline  # Original TP2 configuration
./spark-ctl.sh tunnel            # Keep open for localhost:8000/v1
```

Profile switches refuse active requests, save old container receipts, restart
both ranks, and wait for readiness. `start` reuses the selected profile.
Both DAC interfaces per host now use persistent MTU9000. 8972-byte DF pings
passed both ways on both subnets. Management-interface MTUs were not changed.

Alternatively: `ssh spark-r0 'bash ~/builds/qwen-tp2/cluster.sh status'`.
Other commands: start, stop, wait, logs, logs-r1. Start is detached and wait
only waits for readiness; closing SSH does not terminate the model.

## Upstream review

Fetched KK `1794dcf18` and b12x master `a7d7d29b` during bring-up. No new
Qwen HC/TP2 collective implementation was found relative to the image.
Recent additions primarily serve DeepSeek and IQ2/Q8 models. b12x `2fca4df8`
adds compiled-cache integrity checking. `555aaa83` absorbs small-page fragments
in standalone benchmark runners; this does not automatically affect vLLM.
The pinned serving image has not been modified or upgraded on that basis.


## Bring-up receipts

The checkpoint was copied over 10.100.168.0/24 with rsync and compared using
`rsync -nrc`; the comparison reported no differing files. The original r0
checkpoint was moved, not duplicated, under builds. Both image IDs match
`sha256:688d27699c48c827e7375c51e79f125161187fddfbabd1bc343521986fc4f768`.

Both RDMA test ranks passed all-reduce checks for 1, 1024, and 1048576 FP32
values. NCCL selected both direct RDMA interfaces; Qwen startup separately
confirmed RoCEnante with world size 2. The first unpaired RDMA attempt timed
out while the worker image was still downloading; the paired run passed.

The initial vLLM launch hit its default 92% startup free-memory check after
communication initialization. Setting `--gpu-memory-utilization 0.80` allowed
startup; the explicit KV budget remains 20 GiB per rank and overrides automatic
KV sizing. This is not a reduction to context, precision, or the configured KV
pool. The engine reports 2,229,941 shared logical KV tokens; full pool saturation
has not been tested.

The long-context probe returned all three passkeys in 119.9 seconds, including
prefill and 37 generated tokens. This is a functional check, not a decode-speed
benchmark or a full quality evaluation. Resident PLE is confirmed in both ranks'
resolved configuration (`table_memory='device'`, `cpu_offload=False`).


## Current selection — September 26 UTC

`balanced` selects `hc-adaptive+cg4+m5500h`; KV is pinned at 24 GiB per rank for
all profiles. Profiles compose as `<base>+<modifier>`: `cg4` captures every MTP3
verify size (multiples of 4 up to 64) instead of padding odd concurrencies;
`m5500h` serves the step-5500 hybrid checkpoint (`model-5500h`); `fp32ssm` stores
recurrent state in FP32. Full record: [campaign](optimization/CAMPAIGN-20260926.md).

| Metric | Previous default (Sep 24 cfg, re-measured) | Selected |
|---|---:|---:|
| Prefill 8K / 32K / 64K tok/s | 3718 / 3637 / 3405 | 3671 / 3356 / 3349 |
| C1 tok/s 8K / 32K / 64K | 59.4 / 60.7 / 61.9 | 65.1 / 55.0 / 61.8 |
| C8 aggregate 8K / 32K / 64K | 231.5 / 230.8 / 225.0 | 231.5 / 227.7 / 226.9 |
| C3 / C5 aggregate at 8K | 127.0 / 173.9 | 132.0 / 179.0 |
| C16 aggregate 8K / 32K | 342.6 / 332.0 | 344.6 / 328.9 |
| Wiki NLL · GSM8K · MMLU-Pro | 1.396 · 97.6% · 67.2% | 1.397 · 96.8% · 66.9% |
| KV tokens (max-context sessions) | 2,229,941 (8.5×) at 20 GiB | 2,675,242 (10.2×) |
| Peak used RAM r0 / r1 | 97.3 / – GiB | 102.5 / 99.5 GiB |

Speed and quality are equal within measured noise (single LIL cells vary ±3–5%;
C1 moves with MTP acceptance). The selection is a capacity and checkpoint upgrade,
plus a small mid-concurrency decode gain from exact graph sizes. Rejected with
data: b12x beta `e39b437b` (−15% 32K prefill, −7% 64K C8), `nightly-20260926`
(flat), b12x GDN kernels (slowest), published step-5500 (−14% C1 steps with BF16
attention and a Marlin drafter; no quality gain).

Eight concurrent sessions, each a unique 258,000-token context, retrieved 24/24
passkeys at 5/50/95% depth; a follow-up on all eight at once decoded 8/8
concurrently with 95.3% prefix-cache hits and 74.3% peak KV usage, 0 preemptions.

Identical requests are not bitwise reproducible on this stack (per-token
log-probabilities drift ~0.2 nats between repeats, in every profile including the
stock baseline); see the campaign's open issue.

## Selection — September 24 UTC

`balanced` selects `hc-adaptive` at the user's request; `original` selects `hc`.
Both keep identical precision, context capacity, resident PLE and MTP3. The
results below retain the measurements and their uncertainty for both profiles.

## Adaptive HC — selected default

**Historical trial:** two additional 32K/C8 runs measured 224.39 and
221.31 tok/s, versus the initial 230.86. This overlaps the previous HC result
(222.60); a repeatable decode gain has not been established. See the
[repeatability audit](optimization/HC-SEPARATION.md).

| Context | Prefill tok/s | C1 tok/s | C8 aggregate tok/s |
|---|---:|---:|---:|
| 8K | 3739 | 60.06 | 224.90 |
| 32K | 3488 | 59.18 | 230.86 |
| 64K | 3412 | 61.78 | 219.30 |

Against the previous HC run, prefill and C1 engine-step speed are approximately
preserved; C8 token throughput is +2.1%/+3.7%/+0.6%. Single runs and varying MTP
acceptance limit confidence: this is a modest improvement, not a large decode
speedup. Original baseline C8 remains faster at 8K and 64K, with slower prefill.
No new precision reduction, MoE scale change, vocabulary restriction or capacity
cut was introduced. The all-dynamic MoE override was slower and is not enabled.

The optional `coding` profile remains the earlier MTP5/top20 profile; its capped,
thinking-disabled Tetris measurement is a speed workload, not a product-quality
qualification. The everyday default stays MTP3.

To restore the previous HC profile if needed:

```bash
ssh spark-r0 'python3 ~/builds/qwen-tp2/select_profile.py hc && bash ~/builds/qwen-tp2/cluster.sh wait'
```

## Original HC backup full matrix — September 24 UTC

Backup `hc`/MTP3 validation, compared with the original TP2 profile:

| Context | Baseline PP | HC PP | Baseline C1 | HC C1 | Baseline C8 aggregate | HC C8 aggregate |
|---|---:|---:|---:|---:|---:|---:|
| 8K | 3089 | 3737 | 63.37 | 63.47 | 228.73 | 220.18 |
| 32K | 3043 | 3525 | 59.42 | 59.94 | 232.87 | 222.60 |
| 64K | 2880 | 3406 | 65.35 | 54.76 | 232.86 | 218.09 |

HC improves prefill roughly 16–21%, with modestly slower decode engine steps.
Raw C1 rates also vary with draft acceptance. C8 loses roughly 4–6% in these
single runs. Select `baseline` for that tradeoff, or `balanced` for faster
prefill. Neither profile reduces context or changes target quantization.
All six HC cells passed with zero errors and no underfill, warmup timeout,
capacity limitation or detected exact loop. [HC full matrix](../results/tp2/hc-final-matrix.json).

### Original TP2 reference

Unmodified LIL v0.6.2 at `ccd9ad8ced7e387794391bfb0ac6d99b1f66ba6f`.
Same matrix settings as TP1: 30-second windows, max 2048 output tokens,
model-default sampling/thinking, approximate context targeting, C1/C8, cold
standalone client prefill. C8 is aggregate throughput, not per-user speed.

| Requested context | Cold prefill tok/s | C1 decode tok/s | C8 aggregate tok/s |
|---|---:|---:|---:|
| 8K | 3089 | 63.37 | 228.73 |
| 32K | 3043 | 59.42 | 232.87 |
| 64K | 2880 | 65.35 | 232.86 |

All six cells: zero errors, matching effective concurrency, no underfill,
warmup timeout, capacity limit, or detected exact loop. Approximate context
targeting means actual prompt token counts differ slightly from requested
lengths; the unchanged harness records actual lengths.

Official Coding Peak (upstream Sieve prompt, three runs, max 2000 output,
default sampling/thinking): **84.79 tok/s mean**, 85.58 median,
80.72–88.08 range. Previous selected TP1/6.17 averaged 54.75 tok/s.
These are configuration comparisons, not an isolated TP scaling experiment:
TP2 also uses resident PLE, probabilistic/block MTP, BF16 target-head weights,
and kernel 7.0. Single runs do not estimate run-to-run variation.

Peak system used memory (`MemTotal - MemAvailable`) across startup and tests:
**95.55 GiB on r0, 92.71 GiB on r1**. Neither 119 GiB guard fired.
The temporary bring-up guard is not a permanent service memory limit.

Raw results on the Mac: [matrix](../results/tp2/tp2-resident-matrix.json),
[coding](../results/tp2/tp2-resident-coding.json),
[long context](../results/tp2/initial-long-context.json).
On r0 the same receipts and exact Docker/benchmark commands live in `evidence/`.
Repeated `start` preserved both container start times. Status reports both ranks
running and HTTP 200. If only one rank remains alive, run stop then start to
restart the pair; the controller refuses to pretend a partial cluster is healthy.

## Physical cable versus logical interfaces

The original cable uses r0's right QSFP port (f1) and r1's left QSFP port
(f0). Each physical port exposes two logical PCIe/RDMA interfaces. The one-cable mode
selects both HCAs exposed by that port. Their ethtool link speeds
must not be added together as independent physical-link bandwidth.
A second cable exposes the other physical port but shares the same two host-side
PCIe links. Live sysfs reports Gen5 x4 for each of the two PCIe paths. Combined
encoded capacity is about 31.5 GB/s (252 Gb/s) per direction before transaction
overhead, versus a single 200 Gb/s port. This rules out a 2x host-bandwidth gain,
but does not rule out a smaller gain from unused capacity. The final two-cable trial reduced large collective times by 4–6%, with small
and mixed end-to-end changes. Two cables are now selected, using one HCA per
physical cable and PCIe path; listing all four would be truncated to two by
b12x. The launcher restores MTU 9000 on the selected interfaces at startup.
See [the final cable trial](optimization/CABLES.md) for matched LIL results,
limitations, and one-cable fallback. No model precision or context changes were made.
See NVIDIA's topology documentation:
https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html
