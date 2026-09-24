# Diagnostic scripts

Run with `~/venvs/qwen38/bin/python` on Spark. Current serving instructions
are in `../OPERATIONS.md`; measured results are in `../BENCHMARK.md`.

| Script | Purpose |
|---|---|
| `run_trial.py` | Official, unmodified LIL harness: 8k/32k/64k, C1/C8 and cold prefill by default. |
| `tp1_session.py` | Controlled launch/stop with source revisions and settings recorded. |
| `summarize_matrix.py` | Preserve the official result definitions and validity flags. |
| `watch_memory.py` | Finite-duration RAM guard; use `--guard --max-used-gib 119`. |
| `check_images.py` | Five synthetic images; request acceptance check. |
| `check_long_context.py` | 261,888-token passkey retrieval; capacity check, not a full quality evaluation. |
| `cuda_smoke.cu` | Native SM121 CUDA toolkit smoke test. |
| `prune_benchmarks.py` | Preview cleanup by default; `--apply` keeps the two selected completed LIL runs and removes older raw evidence. |

The official loader and Qwen correctness tests live in the upstream b12x
and vLLM checkouts. Superseded local loader tests, profiling helpers and
synthetic kernel experiments were removed at the user's request.

Raw evidence for the retained runs is under `~/bench/tp1-opt-20260922` on
Spark and `../results/tp1-opt-20260922` locally. Source rollback files live
separately at `~/projects/qwen38-rollback/pre-kk-loader-20260923`.

`container_trial.py` starts the digest-pinned eugr nightly with the matched TP1 profile, read-only checkpoint and separate writable compiler caches. Use `--check` to print the command. It refuses to reuse a trial tag or start over an occupied port. `run_trial.py` verifies the image ID, container PID and runtime environment before benchmarking it.

`kernel_617_once.sh` is the reviewed, interactive-sudo entry point for the kernel-only A/B described in `../KERNEL_AB.md`. It adds signed 6.17 packages and schedules one boot without replacing the default kernel or driver userspace. Run only once the active benchmark is finished.
