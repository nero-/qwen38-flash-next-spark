# Performance audit — 2026-09-22 UTC

> Historical investigation record. At the user's request, older raw benchmarks
> and superseded experiment scripts were pruned to retain only the last
> two LIL matrix runs. Use [BENCHMARK.md](BENCHMARK.md) and
> [OPERATIONS.md](OPERATIONS.md) for current results and launch settings.

This records the baseline audit. Subsequent profiling, resident-PLE loader
repair, capacity validation and tuning trials are in
[TP1_OPTIMIZATION.md](TP1_OPTIMIZATION.md).

The current TP1 server already reaches the published MiaAI single-stream
prose rate when tested with comparable short prompts and sampling settings.
No runtime patch, FR-Spec port, restart, or weight change was needed for this
result. It does not imply all coding or long-context requests run at 49 tok/s.

## Verified deployment

- Spark vLLM checkout: `89d4b75f0` (FR-Spec reverted), clean working tree.
- b12x: `0f3a8cbfd1c11d27f04e3ab37a802d522f4f1c68`.
- sparkDash prompt/settings source: `33391e08f0d3421a2b8e13b0f2bd7c03f0434998`.
- Official benchmark: `local-inference-lab/llm-inference-bench`,
  `ccd9ad8ced7e387794391bfb0ac6d99b1f66ba6f`, reports v0.6.2.
- Server: current run-4 settings, NVFP4 draft head, SSD PLE,
  BF16 SSM, FlashInfer prefill, fused CUDA decode, 8,192-token chunks,
  16 sequences, MTP depth 3.
- Model generation defaults: temperature 1.0, top_p 0.95, top_k 20.
  Template defaults to xhigh thinking unless disabled.

## Controlled sampling diagnostic (not the official benchmark)

Source: `diagnostics/matched_bench.py`.
Raw output: `results/sampling-diagnostic-20260922.jsonl` (includes generated
text, API usage, per-position counters and timing).

One short hash-map prose prompt from sparkDash, 600 output tokens, C=1,
seed 42, one 32-token warmup. Forward/reverse variant order over two repeats.
All tests use the same running server. Prefix/page-cache state is warm and
output varies despite greedy sampling, so do not overinterpret small deltas.

| Request settings | Repeat 1 tok/s | Repeat 2 tok/s | Mean |
|---|---:|---:|---:|
| Model defaults, thinking default | 42.34 | 35.11 | 38.73 |
| Temperature 0, top_p 1, thinking default | 49.39 | 42.38 | 45.88 |
| Temperature 0, top_p 1, thinking disabled | 49.45 | 48.58 | 49.02 |

Last row: 3.025 / 2.965 tokens per draft round; 61.19 / 61.06 ms in
server inter-token-latency counters. MiaAI reports 48.7 tok/s, 3.00 tokens
per round and 61.5 ms. This establishes C=1 parity on this diagnostic;
concurrent parity is not established by it.

The default-setting samples emit reasoning, including one that spends the
whole 600-token budget reasoning. The matched samples emit zero reasoning
characters. The historical official matrix also uses much longer prompts
(8k–128k) and duration windows, so sampling is not its only difference from
the MiaAI prose test.

Diagnostic tok/s uses `(usage.completion_tokens - 1) / (last content event -
first content event)`. An SSE event can contain multiple tokens; this client
formula approximates the first-token boundary. Official Coding Peak uses
its own published formula and is recorded separately below.

## Why the previous optimization diagnosis failed

1. **Benchmark mismatch:** the historical matrix's metadata records
   `temperature: null`; its request builder does not disable thinking.
   sparkDash explicitly sets temperature 0, top_p 1, and thinking off.
2. **Invalid coding measurement:** remote `~/bench/coding_bench.py` increments
   its token count once per streamed content event. TTFT starts its timer at
   that same event, producing zero. The checked-in shell version is also
   broken: a heredoc occupies the parser's stdin instead of the curl pipe,
   and its attempted counter counts characters. Neither is Coding Peak.
3. **Unsupported FR-Spec cause:** MiaAI's full-vocabulary vs reduced-head A/B
   reports unchanged acceptance. Its benefit is reduced head traffic. Our
   draft head is already NVFP4, so BF16-head savings and the projected 35%
   gain cannot simply be transferred. A full-head logits mask saves no head
   matrix multiplication and has no demonstrated benefit here.
4. **Concurrent units:** sum of request draft rounds per second is not global
   engine batch steps per second at C>1. The old 18.2 ms at C=8 is derived
   from an aggregate request-round rate; it cannot be compared directly to
   MiaAI's ~131 ms per-stream inter-step latency.
5. **Arithmetic inconsistency:** per-position acceptance 0.73/0.55/0.43 sums
   to 1.71 accepted drafts, or 2.71 tokens including the target token. It
   cannot support a simultaneous 2.2-token mean. Counters must cover the
   same request interval.

## Prefill interpretation

Recorded run 4: 2,473 tok/s at 8k, 2,468 at 32k, 2,150 at 128k.
MiaAI's September 6 table: 2,200 / 2,314 / 2,146. Their 8,192-chunk
profile gives 2,228 / 2,366 / 2,227, with different rope/cache settings.
These do not establish a large prefill deficit on this Spark. They also do
not prove the hardware's maximum has been reached. Optimizing further needs
matched cold-cache tests and profiling, not the FR-Spec acceptance theory.

Do not change application sampling just to obtain a better benchmark number:
it changes the workload/output behavior. Report separate profiles for
thinking/default sampling and greedy/no-thinking traffic.

## Official harness rerun

Command on spark-r0 (no changes to the official harness):

```bash
~/venvs/qwen38/bin/python ~/projects/llm-inference-bench/llm_decode_bench.py \
  --host 127.0.0.1 --port 8000 --model qwen3.8-flash-next-4p89bpw \
  --no-hw-monitor --display-mode plain --no-resume \
  --contexts 8192 --concurrency 1,8 --duration 15 --max-tokens 2048 \
  --standalone-prefill --prefill-contexts 8k,32k,128k --prefill-duration 1 \
  --coding-peak --coding-peak-runs 3 --coding-peak-max-tokens 2000 \
  --output ~/bench/audit-official-20260922.json
```

This intentionally retains model sampling/thinking defaults and uses the
upstream Sieve-of-Eratosthenes Coding Peak prompt. It is not a matched
MiaAI prose benchmark. Coding Peak does not honor `--reasoning-effort` in
this harness version. The new `coding_bench.sh` invokes this official mode,
with optional `CODING_TEMPERATURE` and no custom token counting.

### Measured cold-prefill and default decode

| Nominal prefill context | Actual prompt tokens | TTFT | tok/s |
|---|---:|---:|---:|
| 8k | 8,199 | 3.340 s | 2,455 |
| 32k | 32,161 | 13.104 s | 2,454 |
| 128k | 128,061 | 59.838 s | 2,140 |

One measured request per context; token targeting uses the harness's
estimation mode. These measurements reproduce run 4 within ~1%.
“Cold” here is the harness's cold-prefix test, not a flushed OS page cache.

At nominal 8k, default sampling/thinking, 15-second sustained windows:
C=1 **31.40 tok/s**, C=8 **117.74 aggregate tok/s**, with acceptance
1.911 / 2.127 tokens per draft round, no errors or detected output loops.
Short windows and stochastic outputs limit small-delta interpretation.

### Official Coding Peak

Three completed sequential Sieve requests, model defaults (temperature not
overridden, thinking enabled), 2,000-token limit; all finished with `stop`:

| Run | Completion tokens | Generation tok/s |
|---|---:|---:|
| 1 | 690 | 47.52 |
| 2 | 1,476 | 44.06 |
| 3 | 1,225 | 44.48 |

Mean **45.36 tok/s**, median **44.48 tok/s**. Includes reasoning and content
tokens. This replaces the invalid historical coding rate; it uses a different
prompt and cannot quantify a speedup over the old analyze_sales workload.
Raw official report: `results/audit-official-20260922.json`.

### Official matched-prose comparison

The official harness's completion-stats mode also supports the short prose
request. C=1: three 600-token requests, **49.18 tok/s mean** (48.69–50.03),
zero errors, zero estimated-token fallbacks. This independently confirms the
sampling diagnostic using the requested benchmark package.

C=8: **19.67 tok/s per stream** averaged over 24 requests (median 19.64),
versus MiaAI's published 20.4 per stream: about 3.6% behind. All 24 requests
completed, with no errors or estimated token counts. This is near the
reference concurrently as well. It does not establish identical performance
at longer context or on different content. Raw files:
`results/audit-matched-c1-20260922.json` and
`results/audit-matched-c8-20260922.json`.

Use the `gen_tok_s.avg` per-request statistic for this comparison. In this
harness mode `aggregate_gen_tok_s` divides total tokens by summed request
generation durations; its 19.64 value is not total concurrent throughput.
We do not present 8 times the per-stream rate as measured aggregate throughput.

Reproduction, separately for `C=1 RUNS=3` and `C=8 RUNS=24`:

```bash
C=1
RUNS=3
~/venvs/qwen38/bin/python ~/projects/llm-inference-bench/llm_decode_bench.py \
  --host 127.0.0.1 --port 8000 --model qwen3.8-flash-next-4p89bpw \
  --no-hw-monitor --display-mode plain --no-resume --completion-stats \
  --prompt 'Write a detailed step-by-step explanation of how a hash map works, including collision handling, resizing, and time complexity. Be thorough.' \
  --profile-concurrency "$C" --profile-runs "$RUNS" --max-tokens 600 \
  --completion-stats-temperature 0 --completion-stats-top-p 1 \
  --reasoning-effort none --completion-stats-correct-regex '' \
  --completion-stats-no-prefill-scout \
  --output ~/bench/audit-matched-c"$C"-20260922.json
```

This mode uses request usage tokens, with the settings recorded in metadata.
Correctness scoring is disabled; 600-token truncation is intentional for this
throughput comparison. It is not a quality evaluation or a long-context test.
At C=8 the harness keeps eight workers supplied with 24 total requests;
MiaAI uses repeated sparkDash waves, so batch-tail behavior may differ.

## Sources

- [MiaAI reference and measured tables](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark)
- [MiaAI September 9 full/subset A/B](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/blob/main/CHANGELOG.md)
- [sparkDash request settings](https://github.com/MiaAI-Lab/sparkDash/blob/main/server/collectors/DecodeBench.js)
- [sparkDash prompts](https://github.com/MiaAI-Lab/sparkDash/blob/main/src/shared/llmPrompts.js)
- [Official benchmark](https://github.com/local-inference-lab/llm-inference-bench)

## Final state and validation

Server health returned HTTP 200 after all runs; zero running/waiting requests.
The vLLM checkout remains clean. Changes are local benchmark tooling,
documentation, and saved evidence. The official harness and serving runtime
were not patched. Shell syntax and Git whitespace checks passed; saved JSON
was parsed and checked for request errors and estimated-token fallbacks.
