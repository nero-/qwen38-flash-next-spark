# Single-Spark optimization experiments — September 22, 2026

> Historical investigation record. At the user's request, older raw benchmarks
> and superseded experiment scripts were pruned to retain only the last
> two LIL matrix runs. Use [BENCHMARK.md](BENCHMARK.md) and
> [OPERATIONS.md](OPERATIONS.md) for current results and launch settings.

Constraints: TP1, native 262,144 context, one active request, MTP 3,
unchanged checkpoint, vocabulary, KV precision and BF16 recurrent state.
This is an experiment record, not a claim that all candidates are faster.

## Baselines

Unmodified `local-inference-lab/llm-inference-bench` at
`ccd9ad8ced7e387794391bfb0ac6d99b1f66ba6f` was used for the earlier audit;
see the exact checkout identity in `PERFORMANCE_AUDIT.md`.
`diagnostics/run_trial.py` invokes the official harness and saves its command,
JSON report and before/after engine metrics. Prose uses five 600-token requests,
temperature 0, top_p 1, thinking off. Coding uses the official Coding Peak
Sieve workload, three runs, max 2,000 tokens, upstream sampling defaults.
The coding invocation also runs a short 8k C1 matrix; those are separate results.

| Configuration | Prose tok/s | Coding Peak tok/s | Engine KV capacity |
|---|---:|---:|---:|
| Original: disk PLE, seqs 16, KV 20 GiB, chunks 8192 | 48.51 | 49.53 | 1,392,826 |
| Control: disk PLE, seqs 1, KV 4 GiB, chunks 2048 | 48.11 | 48.56 | 278,521 |
| Resident PLE, seqs 1, KV 4 GiB, chunks 2048 | 48.77 | 49.00 | 278,521 |
| Original disk profile, OMP threads 1 | 49.16 | 44.59 | 1,392,826 |

These small differences do not demonstrate a win for reducing sequence capacity.
Sampling and MTP acceptance can vary between runs. The original baseline and
control ran before a pause; subsequent resident measurements ran later that day.
Prose engine-counter decode rounds averaged 61.73 ms (original), 61.17 ms
(disk control), and 60.63 ms (resident). Mean acceptance lengths were 2.989,
2.936 and 2.951 respectively. The resident/control difference is under 1% and
has not been established with repeated paired trials.

The one-thread trial had 61.35 ms prose rounds. Across its combined 8k matrix
and coding phase, rounds averaged 62.51 ms versus 62.56 ms originally, while
acceptance length fell from 2.677 to 2.527. Coding output rates therefore do
not isolate a thread-related regression. Its official cold-prefill results
were 2,442 / 2,425 / 2,118 tok/s at nominal 8k / 32k / 128k, also close to the
original audit's 2,455 / 2,454 / 2,140. **No convincing thread-setting win**;
the original 16-thread setting was selected for restoration.

## GPU trace

Eight decode iterations captured with Nsight Systems, separately at C1 and C8.
Profiling runs are diagnostic and must not be used as throughput benchmarks.
Raw trace and SQLite exports are retained on Spark under
`~/bench/tp1-opt-20260922/disk-decode.*`.

At C1, the captured kernel span was 521.06 ms, with 482.72 ms of union GPU
activity and 38.34 ms without a kernel active. Cumulative kernel times include
overlap: target MoE 154.38 ms, other b12x dense 142.73 ms, cuBLAS BF16 WMMA
87.12 ms, full-vocabulary heads 57.51 ms, and PLE GPU kernels 0.62 ms.
The last number excludes SSD/CPU work and does not measure total PLE overhead.
CUDA event synchronization includes waiting for GPU work; it is not all
removable host overhead.

The checkpoint has `ple_layer_ids: [2]`: **one PLE layer**, not 48.
Dense projections and MoE are the principal measured GPU costs.
The BF16 HyperConnection projections are a candidate for native b12x GEMV
screening; a faster synthetic kernel alone would not establish a serving gain.

## Resident PLE bring-up

`VLLM_PLE_TABLE_MEMORY=device` is invalid. The environment accepts `ram` or
`disk`; resident storage is selected by unsetting it and setting
`VLLM_PLE_CPU_OFFLOAD=0`. The first failed attempt was a configuration error,
not evidence of insufficient memory.

The corrected configuration exposed a loader regression from the earlier
mapped-host change: `_copy_embedding_shard` assumed every meta tensor carried
`FileTensorSource`. b12x direct loading instead retains source ownership in
its transfer session. The fix uses the explicit file-read branch only when
file metadata exists; other CUDA destinations reach the existing `copy_weight`
session path. No quantization or model math changes.

The backed-up patch is retained remotely as `resident-loader.patch` alongside
`b12x_ple.py.before`. `diagnostics/test_resident_loader.py` passed all nine
byte-exact cases: session, file-backed and ordinary tensor sources, each with
two partial-overlap cases and a no-overlap case.

Resident trial `resident3` loaded successfully: 99.44 GiB model allocation,
4 GiB KV, 278,521 KV-token capacity, and approximately 9 GiB MemAvailable
during short requests. A 261,888-token prompt completed with 37 output tokens
in 145.44 seconds and correctly retrieved all three passkeys inserted near
12k, 128k and 248k. There were no KV preemptions. This is a capacity and simple
retrieval check, not a comprehensive quality evaluation or official prefill
benchmark. The response and usage are in `resident3-long-context.json`.
MemAvailable reached a minimum of 6.91 GiB across startup, benchmarks, kernel
screening and the long-context request; the 3 GiB memory guard never triggered.
Swap used started at 0.141 GiB, peaked at 1.588 GiB and ended at 0.151 GiB;
this was not a zero-swap experiment.

Resident PLE has **no demonstrated decode win** in these measurements. It
remains a working optional configuration, not the recommended speed upgrade.

## BF16 projection screening

Synthetic BF16 inputs tested both HC shapes at M=1 and M=4 with FP32-reference
checks, fixed allocations, graph capture/replay and alternating timed arms.
Each measured replay was preceded by a 128 MiB cache flush. Raw samples,
kernel-source hash and GPU snapshots are in `hc-projection-screen.json`.

| M | N × K | Torch median µs | b12x SIMT µs | b12x MMA µs |
|---:|---|---:|---:|---:|
| 1 | 336 × 10240 | 61.44 | 56.37 | 190.18 |
| 1 | 10240 × 320 | 54.27 | 49.89 | 55.58 |
| 4 | 336 × 10240 | 60.56 | 59.39 | 192.51 |
| 4 | 10240 × 320 | 59.09 | 56.16 | 57.04 |

These measurements include graph replay/event timing overhead and use
synthetic inputs while an idle server remains resident. They do not establish
end-to-end gains. SIMT's small candidate gains require proper preparation
integration and model evaluation before promotion; no HC kernel change was
installed. MMA was substantially slower on the narrow projection.

## Reproducibility and configuration at the end of the first round

`serve.sh` now defaults to the measured disk profile: 262,144 context, 20 GiB
KV, 16 sequence capacity, 8,192-token chunks, MTP 3, BF16 recurrent state,
FlashInfer GDN prefill and CUDA GDN decode. The earlier wrapper still used
65,536 context and 1.5 GiB KV and omitted the GDN/BF16 overrides. Its
`--check` now prints the command locally instead of forwarding an unsupported
flag to vLLM. Bash syntax and the remote dry-run were checked.

The resident loader fix is the only vLLM source change; b12x is unchanged.
No HC kernel replacement, vocabulary restriction, precision reduction or
speculative-depth change was installed. Benchmark wrappers do not modify
the official harness. Failed attempts and all raw trial artifacts remain
under `~/bench/tp1-opt-20260922` on Spark.

The original profile is restored as `final-baseline`; `/health` and the
thinking-disabled `OK` completion both passed. No benchmark or memory-guard
process remains running. `smoke.sh` now uses the existing virtualenv and
checks the actual response instead of accepting an empty thinking-only result.

Historical raw evidence from this first round was selected for deletion
under the two-run retention policy. The old local PLE patch and source
revision are preserved separately for rollback at
`~/projects/qwen38-rollback/pre-kk-loader-20260923`.

To reproduce the resident option, use the upstream launcher directly after
stopping the existing server; `serve.sh` deliberately defaults to disk PLE:

```bash
cd ~/projects/vllm
env -u VLLM_PLE_TABLE_MEMORY \
  VLLM_PLE_CPU_OFFLOAD=0 B12X_POLICY_MODE=auto OMP_NUM_THREADS=16 \
  MODEL_PATH="$HOME/models/Qwen3.8-Flash-Next-NVFP4" \
  MAX_MODEL_LEN=262144 MAX_NUM_SEQS=1 MAX_NUM_BATCHED_TOKENS=2048 \
  KV_CACHE_MEMORY_BYTES=4294967296 NUM_SPECULATIVE_TOKENS=3 \
  CUDAGRAPH_CAPTURE_SIZES=auto \
  bash serve-qwen38-flash-next-nvfp4-tp1.sh -- \
    --mamba-ssm-cache-dtype bfloat16 \
    --gdn-prefill-backend flashinfer --gdn-decode-kernel cuda
```

## Other reference optimizations

The sparkring TP2 profile includes projection overlap and compact MTP options
already present in this fork; compact MTP is activated for the single-sequence
trial. RoCE and TP-specific sharding require multiple devices. Prompt/recurrent
cache reuse is workload-dependent and does not establish higher sustained
decode speed. No TP2-only changes or reduced vocabulary were applied.

## CUDA 13.3 and loader investigation (September 22 HST / September 23 UTC)

The user installed `cuda-toolkit-13-3` 13.3.1 alongside 13.0. Explicit test
launches use `/usr/local/cuda-13.3` and its `ptxas`; nvcc reports 13.3.73.
`diagnostics/cuda_smoke.cu`, compiled for SM121, passed with runtime 13030
and driver API 13000. The driver remains 580.178.04; PyTorch remains
2.13.0+cu132 with its packaged CUDA 13.2 libraries. A toolkit upgrade does
not replace those libraries. No driver or PyTorch upgrade was performed.

The installed b12x commit `0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68`
selects `managed` in `loader/_pool.py::weight_allocation` on this Spark:
both pageable-memory and host-page-table capabilities were confirmed true.
This is distinct from PLE placement: switching PLE between disk and RAM did
not test the allocation policy for the rest of the model. Fetched upstream
master `4f3028b19c1d8290dc72b6f483aba40de23eae5a` had no loader changes
relative to the installed source. It was not installed during this A/B.

InstantTensor 0.2.0 was installed without changing other dependencies.
The trial changes `--load-format` only, retaining b12x kernels, MTP 3,
full vocabulary, BF16 recurrent state, FP8 KV and the same checkpoint.
Both arms explicitly use CUDA 13.3, disk PLE, 20 GiB KV, 16 sequences and
8,192-token chunks. These experiments use the official harness through
`diagnostics/run_trial.py`; `--cases prose` can run just the matched prose
probe. Engine-counter decode-round latency is recorded separately from
tokens/s because MTP acceptance changes the latter.

The earlier `resident8k` trial exceeded the user-selected 119 GiB system-use
ceiling, reaching approximately 119.2 GiB; its guard terminated the server.
Only prose completed (49.19 tok/s). Coding and a full-context request did
not complete, so that resident configuration is not validated. The later
`managed4k` attempt was cancelled before benchmarking to wait for the CUDA
installation. The user authorized disk PLE for this testing round.

The upstream launcher's explicit image limit was **one**, not four.
Appending `--limit-mm-per-prompt '{}'` restores vLLM's default 999 items per
modality; context and memory still constrain requests. `managed-cu133`
accepted five synthetic 224×224 images (HTTP 200) and correctly answered
`5`. The fixture and captured response are retained; this checks request
acceptance, not comprehensive visual accuracy.

The QAD checkpoint's published model card does not report a KLD figure.
No KLD was measured here and no EXL3 conversion was applied. Quality
preservation in this investigation means no additional precision, vocabulary
or context reduction relative to the existing deployment; it does not claim
the existing quantization is identical to BF16.

### Benchmark method selected by the user

The preferred comparison is now the upstream LIL sustained-decode matrix,
not the MiaAI-style short-prose workload: contexts 8k/32k/64k, concurrency
1/8, 30-second measured windows, max 2,048 generated tokens, and standalone
cold prefill at those same context lengths. No temperature or reasoning
override is supplied. `diagnostics/run_trial.py` defaults to this matrix;
the old prose and Coding Peak probes remain explicit optional cases.

The preliminary prose A/B was 48.90 versus 53.80 tok/s (managed versus
InstantTensor), with 60.78 versus 55.76 ms decode rounds. An InstantTensor
repeat returned 54.25 tok/s. Coding Peak was 45.45 versus 53.82 tok/s, while
combined matrix/coding rounds were 61.93 versus 56.94 ms. These figures are
retained as separate diagnostics, not substituted for the requested matrix.
The harness did not retain full prose text, so those reports cannot establish
output equality. Ten byte-preservation tests passed, including ordinary,
file-backed and b12x-session PLE paths plus InstantTensor's packed uint8 and
BF16 transport with storage retained after loader closure.

### End state and upstream handoff

The requested InstantTensor LIL matrix completed; its authoritative table and
raw links are at the top of [BENCHMARK.md](BENCHMARK.md). The user then asked
to clean up and stop further experiments. No second managed-loader matrix or
additional kernel/allocator changes were made. The API remains healthy on
InstantTensor + CUDA 13.3 toolkit, disk PLE, 256k max context and MTP 3.
No benchmark or memory-guard process remains. The tested launcher is installed
as `~/projects/qwen38-flash-next-spark/serve.sh`; Mac instructions are in
[OPERATIONS.md](OPERATIONS.md). Peak memory over the whole InstantTensor
session was 117.59 GiB, including the separate loader tests, with no guard
trigger and no increase from the initial ~0.00015 GiB swap usage.

A final **fetch only** found these new upstream commits, not applied locally:

- [`1ec67ef`](https://github.com/local-inference-lab/b12x/commit/1ec67ef61104b33c309166dc39254fb5279882a6): ordinary CUDA weights via a bounded 16 MiB asynchronous io_uring/copy ring; removes the custom allocation hooks.
- [`ae43118`](https://github.com/local-inference-lab/b12x/commit/ae4311820c82f5b8b7e2590b829d6244402d526c): invalidates old autotuning selections for the new allocation layout.

The loader commit records DSV4.1 TP4 validation with a matching vLLM
allocation-hook cleanup. This is not Qwen3.8 TP1 validation and does not
establish a 2× serving improvement here. Next session: inspect matching vLLM
changes, preserve/reconcile the local PLE patch, validate loader bytes and
quality, allow retuning as required, then rerun the exact saved LIL matrix.
Do not apply the new b12x alone without checking integration compatibility.

## September 23: upstream loader update and container trial

Updated isolated source branches to vLLM `cff8aebfe` and b12x `0332cc5`. The former local PLE patch was not reapplied; the upstream loader handles that route. Targeted checks: 63 b12x loader tests, 74 vLLM loading/config/transfer tests, and 115 model/QSA tests passed; 9 skipped. The six-cell official LIL matrix completed without errors. MTP-normalized throughput remained approximately unchanged from InstantTensor (−1% to +1.4%); no 2× speedup appeared on TP1. Peak system memory during the guarded source session was 112.64 GiB. Five images succeeded, and a 261,888-token prompt returned all three passkeys.

The requested `eugr/spark-vllm-b12x:nightly-20260923` trial uses the pinned image and isolated caches described in BENCHMARK.md. No host CUDA, PyTorch, checkpoint, or source checkout is replaced. It uses the source baseline's MTP 3, BF16 state, full vocabulary, 20 GiB KV, and disk PLE. The first launch exposed a non-root Triton cache permission issue before weights loaded; explicitly placing Triton/CUDA/config caches in the writable cache mount fixes that launch issue.

The matched image/CUDA/BF16 matrix showed unchanged prefill and normalized decode from −4.1% to +0.9% versus source. Official Coding Peak measured 50.69 tok/s on the image and 52.50 on the updated source build (three runs each, default sampling/thinking, MTP 3). GPU checks found P0 at 2496 MHz, no active thermal/power slowdown, and one engine. These checks do not explain or reproduce the reported 60→90 coding rate. The earlier 90–95 Tetris claim in the conversation explicitly referred to TP2; the newer quote's hardware and exact workload remain unverified.

A standalone b12x decode/BF16 experiment was rejected by vLLM's backend resolver: it requires b12x prefill and decode together. Although the decode kernel accepts BF16, its paired prefill requires FP32. The follow-up uses both b12x paths with FP32 state, preserving MTP 3 and increasing state precision. It still reserves 20 GiB KV and reports 1,085,812 KV tokens.

The supported b12x/FP32 image test completed all six LIL cells without errors, but prefill was approximately 5% slower and Coding Peak was 51.42 tok/s. Keep the CUDA/BF16 profile for the next kernel-only comparison. The complete session peaked at 112.55 GiB with no guard trigger. Kernel 7.0.0-1019 is active; the 6.17 one-time test is prepared in KERNEL_AB.md and awaits interactive sudo.
