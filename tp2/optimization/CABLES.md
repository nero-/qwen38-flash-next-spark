# Final second-cable trial — 2026-09-24

The model comparison keeps the pinned image, checkpoint, adaptive HC, MTP3,
resident PLE, BF16 recurrent state and target head, FP8 KV, context limit and
batching settings fixed. No model precision or vocabulary change is involved.

## Wiring and interface selection

Cable 1 connects r0 f1 to r1 f0. Cable 2 connects r0 f0 to r1 f1.
Each physical port exposes two interfaces backed by the two shared PCIe paths.
The installed b12x `discover_hcas()` truncates explicit lists to two entries;
listing all four would not make b12x use all four.

| Mode | r0 HCA list | r1 HCA list |
|---|---|---|
| One cable | rocep1s0f1,roceP2p1s0f1 | rocep1s0f0,roceP2p1s0f0 |
| Two cables | rocep1s0f1,roceP2p1s0f0 | rocep1s0f0,roceP2p1s0f1 |

The two-cable trial therefore keeps two rails, one per PCIe path, but assigns
them to different physical cables. Both selected interfaces use MTU 9000.
GID index 3 on the new rail is its existing IPv4 link-local address, exchanged
by the backend; the old rail retains its static address. Added temporary
10.100.171.2/24 and .1/24 addresses allow a successful 8972-byte DF ping.
They were not the selected RDMA GIDs and were removed after testing. Original interface state is saved in
`results/tp2/cables-network-r{0,1}-before.json`.

## Communication result

Four alternating runs: one cable A, two cables A, one cable B, two cables B.
Each uses the unchanged upstream b12x collective benchmark, 10 warmups,
40 samples, two interleaved blocks, and 10 operations per graph replay.
All reduction correctness gates and bit-exact all-gather checks passed.
Port counters confirm both physical cables carry traffic in the two-cable
case, and the excluded RDMA interfaces have zero data-counter increments.

Means of the two run medians (microseconds; lower is better):

| Graph operation | One cable | Two cables | Time reduction |
|---|---:|---:|---:|
| All-reduce, 8 KiB | 9.30 | 9.15 | 1.6% |
| All-reduce, 64 KiB | 15.35 | 14.65 | 4.6% |
| All-reduce, 1 MiB | 70.75 | 66.60 | 5.9% |
| All-reduce, 2 MiB | 126.80 | 118.70 | 6.4% |
| All-gather, 4 × 124160 BF16 values/rank | 70.65 | 66.70 | 5.6% |
| All-gather, 16 × 124160 | 241.15 | 228.20 | 5.4% |
| All-gather, 64 × 124160 | 1179.00 | 1127.35 | 4.4% |

This is a repeatable modest collective improvement, not doubled bandwidth
and not yet a measured model-throughput improvement. These timings include
collective/kernel work; they are not a standalone line-rate measurement.
Raw receipts: `results/tp2/cable{1,2}-{a,b}.json`, traffic counter receipts,
and `cables-comm-summary.json`. Launcher: `comm_cables.py`.

## Model comparison

Unmodified LIL harness `ccd9ad8ced7e387794391bfb0ac6d99b1f66ba6f`, 30-second
measurement windows, identical matrix commands. One model matrix per mode;
these do not establish confidence intervals. Docker image and model arguments
match exactly; only `NCCL_IB_HCA` and `B12X_ROCE_HCA` differ between ranks'
recorded environments. Twelve decode cells passed with zero errors, no
underfill, warmup timeout, capacity limitation or detected loops.

| Context | PP: one → two | C1 TG: one → two | C8 aggregate TG: one → two |
|---|---:|---:|---:|
| 8K | 3777 → 3738 | 65.07 → 62.61 | 223.46 → 227.46 |
| 32K | 3563 → 3644 | 61.54 → 67.21 | 227.40 → 235.20 |
| 64K | 3393 → 3409 | 62.52 → 61.65 | 224.42 → 216.56 |

All values are tokens/s. LIL approximate context targeting gives prompt
lengths of approximately 8.2K/32.2K/64.1K for standalone prefill, not exact
requested lengths; the two runs differ by one prompt token.

Step-rate changes (8K/32K/64K): C1 **+0.98% / −0.31% / +0.33%**;
C8 **+0.50% / +3.88% / +0.82%**. The 32K C1 raw +9.2% is explained by
acceptance length rising 2.309 → 2.529 while step rate stays flat. Likewise,
the 64K C8 raw −3.5% coincides with acceptance falling 2.367 → 2.266,
while step rate rises 0.82%. Do not label either raw difference a cable effect.

Conclusion: repeatable modest collective improvement; mixed prefill and raw
TG, with mostly small step-rate changes. No demonstrated large model speedup.
**Keep two cables selected**, because the communication improvement is
repeatable and the available cable can be used without changing model quality
settings. This is a configuration preference, not proof of a
statistically significant end-to-end win. Adaptive HC/MTP3 remains the default;
original HC remains the profile backup. No four-HCA backend patch was attempted.

Raw model matrices, exact commands, metrics, rank inspections and the generated
`cables-model-summary.json` are in `results/tp2/`. Run `summarize_cables.py`
to regenerate the comparison. Memory monitoring covers both runs: peak used RAM was 96.38 GiB on r0 and 93.50 GiB on r1, with no swap use or guard trigger (119 GiB ceiling).

## Operation and fallback

Both hosts have `~/builds/qwen-tp2/selected-cables.txt` set to `2`. The launcher
checks both selected links and GID mappings before starting. If a selected
interface has reverted to MTU 1500 after reboot, it sets MTU 9000 through a
short-lived root Docker container with NET_ADMIN and read-only mounts of the
host `ip` binary and libraries. This uses the Docker access already granted to
nero. It changes only the selected DAC interfaces, not persistent network files
or management interfaces. No extra IP addresses are needed on cable 2: the
backend exchanges each interface's existing IPv4 link-local GID.

The final startup check restores the new interfaces to MTU 1500 while both
model ranks are stopped, then verifies the launcher restores 9000 on each.
The Mac controller and profile aliases are unchanged.

To return to one cable, stop both ranks with `cluster.sh stop`, remove the two
stopped `qwen-tp2` containers, write `1` to `selected-cables.txt` on both hosts,
and run `cluster.sh start` followed by `cluster.sh wait` on r0. Do not change
one rank independently. Both old interfaces retain their persistent MTU 9000
configuration. Cable 2 can remain physically connected in one-cable mode.

A four-interface b12x patch would still share the same two PCIe paths. It might
change utilization or overhead, but cannot double host bandwidth, and has not
been tested. This final trial is complete; no further optimization runs are
scheduled by this work.

Final verification: both ranks running, HTTP 200, and a greedy non-thinking
request returned the requested `OK` after startup with only the original
link-local addresses on cable 2. Temporary memory monitors are stopped.
