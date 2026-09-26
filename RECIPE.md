> **TP2 deployment:** Both Sparks use resident PLE and the `hc-adaptive+cg4+m5500h` MTP3 default on the step-5500 hybrid checkpoint with 24 GiB KV (`previous` = September 24 `hc-adaptive`; `hc` is the backup). See the [September 26 campaign](tp2/optimization/CAMPAIGN-20260926.md).

Final cable trial (2026-09-24): **two cables selected, adaptive HC/MTP3 unchanged**. Large collective times improved 4–6%; model results were modest and mixed. [Measured comparison and one-cable fallback](tp2/optimization/CABLES.md).
> Use [the TP2 operations guide](tp2/README.md) and `~/Agent/Builds/spark-ctl.sh` on the Mac.
> The TP1 instructions below are historical; do not start them alongside TP2.

# Recipe & Postmortem — Qwen3.8-Flash-Next NVFP4 (QAD) on DGX Spark (TP1)

> **2026-09-22 audit:** Several interpretations in this historical postmortem
> were incorrect. The FR-Spec acceptance explanation and projected +35% gain
> are withdrawn; the custom coding benchmark's 16.4 tok/s counted chunks.
> Controlled short-prose requests on the unchanged server already measured
> 48.58–49.45 tok/s. The official harness then confirmed 49.18 tok/s C=1
> and 19.67 tok/s per stream C=8, plus 45.36 tok/s built-in Coding Peak.
> See [PERFORMANCE_AUDIT.md](PERFORMANCE_AUDIT.md).

> **Latest configuration:** This is the historical recipe. CUDA 13.3 toolkit,
> loader comparisons, resident-PLE capacity findings and the image-cap change
> are recorded in [TP1_OPTIMIZATION.md](TP1_OPTIMIZATION.md). Use
> [OPERATIONS.md](OPERATIONS.md) for the current launch command.

**Status:** Stable and serving. This document records what we built, what the
numbers are, every optimization attempted, which ones worked, and — in detail —
the one we could not land (FR-Spec reduced draft vocabulary), including the
exact blocker, so a future attempt (human or agent) can pick it up without
re-deriving two days of findings.

---

## 1. Final working configuration

**Host:** DGX Spark (GB10, SM121, 121 GiB unified LPDDR5X, CUDA 13.0, driver 580.178.04)
**Serving stack:** vLLM fork `local-inference-lab/vllm` @ `dev/karmic-kraken`
(`af9e4dc`) built from source + `b12x` 1.3.0 built from source (the PyPI wheel
lacks the SM121/dense-activation changes this integration requires — the fork's
own `docs/features/quantization/b12x.md` says so).
**Checkpoint:** `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` — 98.6 GiB,
quantization-aware distillation (QAD): NVFP4 routed experts + PLE tables,
MXFP8 attention/shared experts, W4A16 NVFP4 vision FC2/MTP, BF16 embeddings.

### Launch command (the whole recipe)

```bash
ssh spark-r0
cd ~/projects/vllm
rm -f ~/serve_qwen38.log ~/serve_launch.log
nohup env \
  PATH="$HOME/venvs/qwen38/bin:/usr/local/cuda/bin:/usr/bin:/bin" \
  VLLM_PLE_TABLE_MEMORY=disk \
  B12X_POLICY_MODE=auto \
  MODEL_PATH=$HOME/models/Qwen3.8-Flash-Next-NVFP4 \
  LOG_FILE=$HOME/serve_qwen38.log \
  MAX_NUM_SEQS=16 \
  KV_CACHE_MEMORY_BYTES=21474836480 \
  MAX_MODEL_LEN=262144 \
  MAX_NUM_BATCHED_TOKENS=8192 \
  CUDAGRAPH_CAPTURE_SIZES=auto \
  bash serve-qwen38-flash-next-nvfp4-tp1.sh -- \
    --mamba-ssm-cache-dtype bfloat16 \
    --gdn-prefill-backend flashinfer \
    --gdn-decode-kernel cuda \
  > $HOME/serve_launch.log 2>&1 &
```

Model name for clients: `qwen3.8-flash-next-4p89bpw`. Startup ~5 min
(b12x warmup + graph capture). Full details: `OPERATIONS.md`.

### Why each knob exists (measured, not guessed)

| Knob | Value | Measured effect |
|---|---|---|
| `VLLM_PLE_TABLE_MEMORY=disk` | SSD io_uring row reads | Frees 26.8 GiB for KV. Original profile supports 1.39M KV tokens. A matched small-capacity control differed from resident by only 0.54 ms per decode round; see `TP1_OPTIMIZATION.md`. |
| `MAX_NUM_SEQS=16` | concurrent streams | At 8: 116.7 tok/s. At 16: **162.2 tok/s.** Must pair with full graph coverage. |
| `MAX_MODEL_LEN=262144` | native context | The served profile; YaRN 512k exists but untested here. |
| `MAX_NUM_BATCHED_TOKENS=8192` | prefill chunk | Part of the run-4 prefill gain (2,473 tok/s @8k vs 2,114 at 2048). |
| `CUDAGRAPH_CAPTURE_SIZES=auto` | full graphs 4–128 tok | seqs(16) × MTP width(4) sets ceiling 128 → every verify batch (16×4=64 tok) has an exact graph. **Required for C=16.** |
| `--mamba-ssm-cache-dtype bfloat16` | SSM state BF16 | +5–12% decode (bandwidth halved). |
| `--gdn-prefill-backend flashinfer` | | **Required with bf16 SSM** — b12x `gdn_prefill` kernel hard-asserts FP32 state (`_impl.py:64`). FlashInfer qualifies on SM12x + 128 heads + CUDA 13. |
| `--gdn-decode-kernel cuda` | fused decode | The b12x decode kernel is also FP32-only; the fused CUDA kernel accepts BF16. Pairs with flashinfer prefill (b12x prefill+decode must be selected together). |

Resulting pool: **KV 1,392,826 tokens**, concurrency 5.31× at 262k context.

---

## 2. Results vs the reference (MiaAI-Lab single-Spark recipe, same hardware)

| Metric | MiaAI (2026-09-06 sweep) | Ours (run 4) | Delta |
|---|---|---|---|
| Prefill @8k | 2,200 tok/s | **2,473** | **+12%** |
| Prefill @32k | 2,304 | **2,468** | **+7%** |
| Prefill @128k | 2,146 | 2,150 | parity |
| KV pool | ~992k tok | **1,392,826** | **+40%** |
| Decode C=1 | 48.7 tok/s (accept 3.0 → 16.2 steps/s) | 33.3 (accept 2.2 → 15.1 steps/s) | different workloads; not a matched comparison |
| Decode C=8 | 162.9 (accept ~2.8) | 116.7 (accept 2.13 → 54.8 steps/s) | −28%; attribution not established |
| Decode C=16 | not measured by them | **162.2** | matches their C=8 |
| Historical coding counter | — | 16.4 chunks/s | **Invalid token-rate measurement** |

**Corrected interpretation:** these benchmark workloads and sampling settings
differ. The observed acceptance gap does not identify reduced vocabulary as
the cause. Matched short-prose requests now reach the reference C=1 rate.

---

## 3. What we changed from the naive install (the actual recipe)

The branch's own `serve-qwen38-flash-next-nvfp4-tp1.sh` ships conservative
defaults. The tuned profile differs in six ways, in order of impact:

1. **PLE table offloaded to SSD** (`VLLM_PLE_TABLE_MEMORY=disk`). The naive
   config keeps the 26.8 GiB n-gram table on-device, which collapses KV to
   1.5 GiB (68k tokens) unless you force KV bytes, and then over-commits the
   unified pool (this is what killed three of MiaAI's servers on 2026-09-04 —
   documented in their repo).
2. **Sequences 1 → 16** with full CUDA graph coverage. The branch default
   (`max_num_seqs=1`) exists to enable an MTP prefill-compaction fast path;
   at 16 you give that up (~2%) and gain 5× aggregate decode.
3. **Context 65k → 256k, batched tokens 2,048 → 8,192.**
4. **BF16 SSM state + FlashInfer GDN prefill + fused CUDA decode.** The
   branch's b12x GDN path is FP32-only; bf16 halves recurrent-state bandwidth.
   This was worth +5–12% decode.
5. **KV pinned explicitly** (`KV_CACHE_MEMORY_BYTES`) rather than via
   `gpu-memory-utilization`. On GB10, vLLM's GMU accounting treats
   MemAvailable as free memory and fills to budget — the over-commit mechanism
   behind MiaAI's server losses. Explicit bytes is the predictable control.
6. **torchvision, flashinfer, cutlass-dsl, ninja on PATH** — four missing
   runtime deps the branch's tp1 script doesn't declare (see install.sh).

Full install script: `install.sh` (idempotent, ~60 min including the 30–60 min
vLLM source build on 20 cores).

---

## 4. Experiments log (what we tried, measured outcomes)

### A. PLE in pinned host RAM instead of SSD — **worked, rejected for TP1**

- **Hypothesis:** the ~6% engine-step gap vs MiaAI (65.4 vs 61.5 ms at C=1)
  was PLE SSD I/O. The old claim of 48 sequential PLE layers was incorrect:
  checkpoint configuration has `ple_layer_ids: [2]`, just one PLE layer.
- **To make it load at all**, we patched the PLE loader: in `mapped_host`
  mode the loading view is a **CPU pinned alias**, but the b12x session
  writer only accepts pool-owned CUDA destinations, and file-backed routing
  was gated to `io_uring` only. Fixes (committed `8796215`):
  - `b12x_ple.py::_copy_embedding_shard` — CPU destination → direct
    `target.copy_(source)` (bypass session writer);
  - file-backed PLE shards re-routed as meta placeholders carrying
    `FileTensorSource`, rows read from the safetensors file directly
    (`_read_file_rows`);
  - `model.py::checkpoint_file_weight_filter` extended to `mapped_host`.
- **Result:** C=1 step **65.4 → 60.5 ms (−7.5%)**, C=8 inverse aggregate request-round rate 18.2 → 17.5 ms (−4%);
  those C=8 values are not per-batch engine step latency.
  These historical timings do not isolate PLE I/O from workload and capacity
  changes. The September 22 Nsight trace instead shows dense projections and
  MoE dominating C=1 GPU time; host waits also include waiting for GPU work.
- **Why rejected on TP1:** pinned mapped memory is *non-evictable committed*
  memory. It leaves only 557k KV tokens (vs 1.39M) and drove MemAvailable to
  ~0. On GB10 the "where" doesn't matter, only the commitment class — and
  pinned consumes the shared unified pool. TP2 behavior must be measured;
  two devices do not by themselves establish that the PLE cost disappears.

### B. Draft-head dtype (NVFP4 vs BF16) — **A/B'd, no difference**

`VLLM_MTP_NVFP4_LM_HEAD=1` (fork default) vs `0`: acceptance median 2.18 vs
2.27, C=1 33.3 vs 32.5 tok/s — noise. Keep NVFP4 (less memory).

### C. FR-Spec reduced draft vocabulary — failed port, benefit unmeasured (§5)

### D. Things measured/ruled out

- **torch.compile** (COMPILATION_MODE): +0.3–1.0%, inside noise — matches
  MiaAI's finding; decode is bandwidth-bound.
- **CUDA graph coverage**: with `seqs=16, MTP=3`, the auto policy captures
  FULL graphs 4–128 tokens; all 20 bench cells ran graphed. Not the gap.
- **K-schedule (speculative depth per batch size)**: their static sweep found
  K=3 optimal at every concurrency; no crossover to exploit at our acceptance.

---

## 5. FR-Spec reduced draft vocabulary — detailed postmortem

**The idea** (from MiaAI): shrink the MTP draft head from 248,320 rows to a
frequent-token subset (we built 51,259 ids from 123 MB of en+code corpus,
100% coverage, all 256 byte-fallback + special tokens pinned). The original probability-concentration explanation and +35% speed projection
were unsupported and are withdrawn. A reduced head can save bandwidth; any
benefit must be measured against our already-NVFP4 head. Correctness requires
proper token-ID/probability mapping through target verification, and has not
been validated for the parked implementation.

**Impact unknown on this fork:** the old half-GEMM/half-acceptance attribution
was not measured. MiaAI's BF16-head byte savings are not our NVFP4-head savings.

### Seven failed launches, each fixing the previous bug

All work preserved on branch `frspec-draft-vocab` (commits `39bb4e3`…
`6870f2f`), with the 51k vocab file at `~/projects/frspec/draft_vocab_65k.txt`.

1. **Shape assert** — checkpoint streams the full-vocab `lm_head.weight` into
   our subset-sized `ParallelLMHead`; `weight_loader` asserts
   `shape == org_vocab_size`. Fix attempt: skip the stream, fill rows later.
2. **Padding mismatch** — head rows pad to 51,264 (multiple-of-64); sliced
   with 51,259 → `copy_` shape error. Fix: zero tail, slice `head_rows`.
3. **Meta tensor (the wall)** — under the b12x loader, streamed weights
   arrive as **meta placeholders** (shape/dtype only, no storage) because the
   loader defers materialization via file-range routing. `clone()`/`copy_`
   impossible. Three workarounds attempted:
   - capture `FileTensorSource` from the placeholder and mmap-read rows →
     **source attribute is stripped** by the loader's re-wrapping;
   - read `lm_head.weight` via checkpoint index + `safetensors.safe_open` →
     **worked** (real CPU tensor: `meta=False shape=(248320,2560)`), but the
     *destination* param was still meta because nothing ever streamed into it
     (we skipped it) — meta params only materialize when a weight loads;
   - `materialize_meta_tensor()` in-place — did not take effect on the
     `ModelWeightParameter` wrapper.
4. **Mask-caching device assert** — a parallel approach (full-vocab head +
   `-inf` mask on non-subset logits) died on an async device-side assert:
   the lazily-built mask cached during one profiling pass collided with a
   different-width logits tensor in a later b12x profile-state release.
5. **Index-vs-value confusion (caught by re-read, not a crash)** — first
   mapping did `draft_vocab_ids[logits]` (gathers logit *values*); correct
   form after masked argmax is identity (target ids come out directly).
6. **Meta + slicing combined** — slice-on-arrival with file-source fallback;
   final crash: `ValueError: FR-Spec: lm_head streamed as meta without a
   file source`. **Definitive root cause:** the b12x loader strips the
   `FileTensorSource` attribute from meta placeholders by the time they
   reach the model's weight iterator.

### The precise blocker (for whoever picks this up)

> **`b12x/loader/_checkpoint.py` / `integration/vllm/loader.py`: meta
> placeholders for non-file-backed weights lose their `FileTensorSource`
> attachment before reaching the model's weight iterator.** Any sliced-vocab
> implementation needs either (a) the loader to preserve the source on meta
> tensors, (b) the model to resolve rows by name via the checkpoint index
> *and* materialize its own parameter (the loader materializes params only
> when a weight streams in — skipping the stream leaves the param meta
> forever), or (c) a "pre-allocated non-meta" parameter type.

Also unresolved: whether a 51,264-row head survives
`Nvfp4OnlineLinearMethod.process_weights_after_loading` (the NVFP4 packing
path was never reached — every attempt died before it).

### Alternative that avoids the whole problem

**Mask on the materialized full head** (keep 248k head, add `-inf` on
non-subset rows before argmax — a registered buffer, built eagerly in
`__init__`, never cached lazily). Does not save head GEMM bandwidth and has no established acceptance benefit. Never landed because the mask
experiment hit bug #4 first; the eager-registered-buffer fix
(commit `26462a0`) should address it and is the *lowest-risk* path if you
want partial gains without touching the load lifecycle. Untested.

### What a successful port needs (checklist)

1. Fix or work around the meta-without-source routing (one of a/b/c above).
2. Verify `Nvfp4OnlineLinearMethod` (or BF16 head — measured equivalent)
   handles 51,264-row packing in `process_weights_after_loading`.
3. Confirm CUDA-graph capture covers the mapping gather (capture-safe: a
   constant int64 table gather).
4. Losslessness test: identical outputs vs full-vocab config on a fixed
   prompt set (temperature 0) — rejection sampling guarantees this, but test
   it (out-of-subset tokens must be *absent*, not degraded).
5. Re-run acceptance and throughput on identical requests; do not assume an
   acceptance increase. Report measured changes, including regressions.

---

## 6. Environment quirks worth knowing (all hit, all solved)

| Issue | Fix |
|---|---|
| `torchvision` missing → registry inspect crash | `uv pip install torchvision==0.28.0 --index-url .../cu132` |
| `flashinfer` missing → Sampler init crash | `uv pip install flashinfer-python` |
| `cutlass` missing → b12x blockscaled import crash | `uv pip install nvidia-cutlass-dsl==4.6.2` |
| "Ninja is required" inside b12x preparation | `uv pip install ninja` **and** put venv bin on `PATH` (the worker subprocess doesn't inherit pip-installed console scripts otherwise) |
| `elfutils/libdwfl.h` not found (deepgemm JIT) | `sudo apt install libdw-dev libelf-dev` |
| `python3.12-dev` missing → CMake can't find Python | `sudo apt install python3.12-dev` |
| Wheels are amd64-only upstream | Everything must build from source on aarch64 (vLLM 30–60 min; b12x a few min) |

---

## 7. Repository map

- `~/projects/vllm` — branch `frspec-draft-vocab`; **serving commit**
  `89d4b75f` (run-4 config; FR-Spec parked on the branch tip `6870f2f`).
  Local patches: PLE mapped_host loading, FR-Spec experiments.
- `~/projects/b12x` — source, editable install.
- `~/models/Qwen3.8-Flash-Next-NVFP4` — checkpoint.
- `~/venvs/qwen38` — runtime env.
- `~/projects/frspec/` — vocab builder + 51k subset file + corpus.
- `~/projects/llm-inference-bench` + `~/venvs/qwen38` — bench tooling.
- `~/bench/*.json` — all raw benchmark data.
- `~/ops-guide/OPERATIONS.md` — day-2 operations.

Docs in the GitHub repo (`Agent/Builds/qwen38-flash-next-spark`):
`README.md` (install/build), `OPERATIONS.md` (day-2 ops), `BENCHMARK.md`
(numbers + all four runs), `TP2_MIGRATION.md` (two-Spark path), `install.sh`,
`serve.sh`, `smoke.sh`, `coding_bench.sh`.

---

## 8. Open ideas for the next agent

1. **Benchmark comparability first** — use the official harness and explicit
   request settings. FR-Spec (§5) is an unmeasured optimization on this fork,
   not a demonstrated +35% win.
2. **TP2** — measure the balance of sharded compute, PLE reads and RoCE
   collectives. `TP2_MIGRATION.md` has the bring-up; gains are not yet measured.
3. **YaRN 512k** — untested; would need the KV arithmetic redone at 2×
   context (§5 of MiaAI's docs covers their approach; same checkpoint family).
4. **max_num_seqs beyond 16** for ≤32k traffic — C=16 ≈ C=8+40%, so there may
   be more; KV at 262k caps full-context concurrency at ~5.3 streams.
5. **b12x upstream**: preserve `FileTensorSource` on meta placeholders — a
   small fix that unblocks sliced-vocab drafting for every b12x user.
