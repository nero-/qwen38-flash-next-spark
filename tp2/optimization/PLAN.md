# TP2 optimization experiments

Selected baseline remains BF16 target head, checkpoint MoE scales, BF16 recurrent
state, FP8 KV, resident PLE, full vocabulary and 256K context. The user authorized
separate MXFP8 target-head/w13-scale tests; passing a small regression suite does
not prove global quality equivalence, so retain an unchanged-precision option.

## Completed preparation

- Persistent MTU 9000 on the two DAC interfaces per host. 8972-byte, DF-enabled
  ICMP passed both directions on both subnets; management interfaces unchanged.
- Exact source image retained. No new upstream Qwen TP2-specific improvement was
  identified in the reviewed KK/b12x heads.
- Upstream RoCEnante benchmark: correctness checks passed; graphed all-reduce
  beats graphed NCCL across 8 KiB through 2 MiB in this run. Keep the current
  custom all-reduce limit while measuring the model. These are microbenchmarks,
  not critical-path fractions of end-to-end inference.
- TP4 source files read from gx10-r0; that deployment was not changed.
- 131 initial GPU checks passed for TP2/TP4 projection partitions and proposal
  distribution, including MTP4/5. Wide-vocabulary replay and row-ownership
  admission added 7 passing checks (138 total) before serving the candidate.

## HC runtime validation

HC-only startup succeeded. Logs confirm 8192-row prefills use 4096 owned rows
per rank, with 96 reduce-scatter and 99 all-gather calls per admitted forward.
The five-image check passed. The 261888-token passkey check retrieved all three
keys in 107.27 seconds (earlier baseline 119.9 seconds; one sample each, not a
controlled throughput estimate). The short functional suite matches both
baseline controls at 5/7: JSON-only formatting and the box-swap logic question
fail in both profiles.

The 235 teacher-forced token log-probabilities vary even across baseline repeat
runs: mean absolute delta 0.12470, maximum 0.88457. HC versus the first baseline
has mean absolute delta 0.12816, maximum 1.26171. These short probes are too
variable and narrow to certify the quantization experiment as lossless.

HC screening (unchanged LIL): cold prefill 3707 tok/s at ~8K and 3366 at ~64K,
versus MTU9000 baseline 3132 / 2882 (+18.4% / +16.8%). C1 decode 60.08 / 62.04
at 8K / 32K versus baseline 64.15 / 58.29; no consistent decode improvement.
Matched seed-42 capped Tetris: 91.20 tok/s versus baseline 89.58, both hit 8192
tokens. Treat that small difference as inconclusive pending repeats.

Subsequent profile caches are separate copies seeded from the stopped previous
profile's keyed caches. Kernel keys still invalidate changed code and shapes;
this avoids recompiling unchanged b12x kernels for every experiment.

## Initial screening results

All rows use MTU9000. C1 cells are individual 30-second LIL windows; Tetris is
one **thinking-disabled** effective-seed-42 sample capped at 8192 tokens. These are screening results,
not confidence intervals. Every listed Tetris output hits the cap.

| Profile | PP 8K | PP 64K | C1 8K | C1 32K | Tetris tok/s |
|---|---:|---:|---:|---:|---:|
| Baseline MTP3 | 3132 | 2882 | 64.15 | 58.29 | 89.58 |
| HC MTP3 | 3707 | 3366 | 60.08 | 62.04 | 91.20 |
| HC + top20 MTP3 | 3709 | 3391 | 60.16 | 66.76 | 88.36 |
| HC + top20 MTP4 | 3707 | 3345 | 62.23 | 59.81 | 94.52 |
| HC + top20 MTP5 | 3700 | 3338 | 56.23 | 62.44 | 99.18 |
| HC + top20 MTP5, MXFP8/w13 | 3725 | 3358 | 56.07 | 54.02 | 98.33 |
| HC MTP5, no top20 | — | — | 55.27 | 54.92 | 95.77 |

The MXFP8/w13 experiment is matched at MTP5 to isolate the two additional
flags. Its short checks pass 6/7 versus matched BF16's 5/7 (JSON formatting
changes to passing; box-swap logic still fails). The 235-token likelihood probe
changes mean NLL 1.74708 -> 1.75721, mean absolute log-probability delta 0.14833,
max 1.67383. This is neither full-distribution KLD nor a sufficient quality
qualification. It also shows no overall performance advantage in this screen,
so BF16 target head and checkpoint MoE scales remain the intended defaults.

## Balanced full matrix

HC/MTP3 full validation: PP 3737 / 3525 / 3406 tok/s at 8K / 32K / 64K;
C1 63.47 / 59.94 / 54.76; C8 aggregate 220.18 / 222.60 / 218.09.
All six cells have zero errors, matching effective concurrency, no underfill,
warmup timeout, capacity limit or detected exact loop. Compared with the initial
TP2 matrix, prefill improves about 21% / 16% / 18%, while C8 is lower by roughly
4% / 4% / 6%. Acceptance-normalized request-round rates are also a few percent
lower. This is a prefill/decode tradeoff, not a universal acceleration.
The original baseline remains selectable for workloads that favor its decode.

## Complete-output coding and profiling

HC/MTP3, same prompt/seed/temperature, output cap raised to 32768: generation
stopped naturally at 15320 tokens, 183.03 seconds, 83.81 generation tok/s.
**Functional coding FAIL:** browser canvas stays blank and the console reports
`SyntaxError: Unexpected token ','`. Node's syntax check agrees at line 21 of
the verbatim extracted script (malformed I-piece rotation data). Keep this
separate from capped speed numbers. No repairs were made to the generated
artifact. See `results/tp2/tetris-complete/VALIDATION.md`.

The user subsequently clarified that Luke measures generation during Qwen's
reasoning while it works on Tetris. The above thinking-disabled runs therefore
do **not** reproduce that workload and must not be used to infer thinking-mode
coding quality. Thinking-enabled LIL runs are recorded separately. The harness
reports whole-response mean generation speed, not an instantaneous reasoning
peak; a capped window is labeled reasoning-only only after checking that its
saved stream contains reasoning and no answer content.

A bounded 8-engine-iteration Torch trace was captured on both ranks, then the
profiler configuration was removed. It includes a 7231-token prefill and seven
decode executions. That odd-sized prefill uses the replicated fallback, so it
does not profile the owned-row HC prefill improvement. Kernel-launch correlation
to decode execution annotations attributes summed kernel times on rank0 to
25.82 ms communication and 199.51 ms other compute; rank1 25.56 / 197.84 ms.
MoE and dense GEMMs dominate the attributed decode compute. Summed durations
double-count stream overlap; these are not end-to-end wall-time fractions.
Some kernels lie outside execute annotations and remain separately classified.
Raw traces and reproducible summaries are under `results/tp2/trace-rank{0,1}`.

## Port scope

Our image already defaults to channel-partitioned HC. The TP4 candidate's
prerequisite is pure-prefill row ownership with full projections, while retaining
channel projections for replicated small batches. This requires porting
`hc_prefill.py`, the model caller, and a default-preserving `reduce_results`
argument to `Qwen3NextSparseMoeBlock`, besides the HC implementation itself.
The TP4 coalesced recurrent-prefill change is deliberately excluded: it is a
separate engine change and absent from this image. No SparkRing transport or
runtime hooks are copied.

Admission is limited to BF16 TP2/TP4, PP1/DP1/DCP1/PCP1, no EP/EPLB/sequence
parallelism/DBO, pure non-captured prefills with >=1024 evenly partitionable
rows. Mixed batches and graph replay retain the replicated-row path. Owned
rows must never enter the decode channel collectives. The target and draft
vocabularies remain full; top-20 filtering applies only to draft proposals and
caches the same filtered pre-temperature distribution for rejection sampling.

Profile order: baseline at MTU9000, HC alone, HC+top20, MTP4, MTP5, and a
separate MXFP8+w13 experiment. MTP graph caps scale with 16 sequences and depth:
64 / 80 / 96. Model/context/KV budgets stay fixed. Candidate source files are
read-only bind mounts with recorded SHA256 manifests; caches are per profile.

## Comparisons

Use unchanged LIL v0.6.2 for 8K/32K C1 screening and 8K/64K cold prefill, then
repeat the full 8K/32K/64K C1/C8 matrix on the selected profile. Record actual
contexts and acceptance; single stochastic samples do not establish small gains.

The first custom Tetris prompt (107 prompt tokens) ran three times at an 8192
output cap: mean 89.85 tok/s, every output truncated mid-JavaScript. It did not
produce a playable game. The recovered TP4 prompt is used for matched coding
measurements, temperature 1, thinking off, effective seed 42. LIL increments its
seed base by run index, so a single measured run uses base 41. The first
post-MTU Tetris sample inadvertently used effective seed 43 and remains labeled
separately in the receipts; it must not be called a seed-42 match.

A larger-cap run on the selected profile will test time to a complete game;
throughput alone is not coding-task completion. Functional output checks and a
small teacher-forced likelihood probe compare each profile against repeated
BF16 controls. This is not full-distribution KLD or a comprehensive quality eval.

## Thinking-enabled Tetris correction and final state

The user clarified that Luke measures reasoning throughput while generating Tetris.
The earlier 8192-token screens explicitly disabled thinking and cannot be compared
to that claim. Unmodified LIL v0.6.2 was rerun with default model thinking enabled
(no `--reasoning-effort none`), the same prompt, temperature 1 and seed base 41.

| Profile | Output cap | Actual output | Generation tok/s | Content |
|---|---:|---:|---:|---|
| HC / MTP3 | 4096 | 4096 | 69.99 | reasoning only, no answer |
| HC / top20 / MTP5 | 4096 | 4096 | 66.62 | reasoning only, no answer |
| HC / MTP3 | 65536 | 27561 | 79.08 | natural stop, reasoning plus answer |

These are single-run averages, not instantaneous peaks. MTP5 did not improve the
matched reasoning window. The complete run took 348.57 seconds of generation.
Receipts: `results/tp2/hc-thinking-window-tetris.json`,
`hc-k20-mtp5-thinking-window-tetris.json`, `hc-thinking-complete-tetris.json`.

The complete thinking-enabled HTML passes JavaScript syntax checking but fails
browser validation: queue drawing throws `TypeError: Cannot read properties of
undefined (reading '0')` at HTML line 799 during initialization (call sites
1022/1023). The board remains blank and animation time stays 0:00. Start, scoring
and pause labels respond to input, but that does not establish a working game.
The original output is preserved unchanged in `results/tp2/tetris-thinking/`.
This single stochastic failure is not evidence that HC reduces model quality;
there is no matched complete baseline generation establishing causation.

Five 672px images passed (answer 5), including the HC-owned prefill branch:
1472 rows, 736 per owner. Memory guards recorded peak used RAM of 97.10 GiB on r0
and 94.88 GiB on r1 throughout this experiment sequence, below 119 GiB. Temporary
guards were stopped after evidence capture. Both model ranks remain on `hc`,
MTP3, BF16 target head, resident PLE, with API health confirmed. The quantized
head/w13 experiment remains unselected. MTU9000 stays enabled.

## Latest HC/kernel follow-up

The provisional `hc-adaptive` selection was reversed after controlled repeats;
selected profile is again `hc`. See
[the bounded isolation and kernel report](HC-SEPARATION.md) for final measurements,
256 passing GPU tests, rejected MoE dispatch override, and rollback details.

## BF16 head kernel screen

A full-vocabulary synthetic kernel probe found 1.40x at one row but only 1.02x
at the MTP3 four-row shape. No implementation was deployed. See
[the operator report](BF16-HEAD.md) for timing, numerical checks and limitations.

## Prefill batch-size audit

[Source review](PREFILL-BATCH-AUDIT.md) confirms MoE receives full prefill rows
after HC gathering; 8K is already above the generic large-tile threshold.
No scheduler-budget change was made or performance improvement claimed.

## User profile preference

User selected adaptive HC as default and original HC as backup. Both remain
MTP3. This supersedes the earlier default-selection decision, not the measurement
uncertainty. Mac aliases: `profile balanced` and `profile original`.

## September 26 campaign

A later campaign selected `hc-adaptive+cg4+m5500h` with 24 GiB KV and recorded
every rejected lever with data. See [CAMPAIGN-20260926.md](CAMPAIGN-20260926.md).
