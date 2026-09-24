# Prefill batch-size source audit — 2026-09-24

No launch changes or benchmarks were performed. The running configuration is
HC/MTP3 with scheduler budget 8192, context 262144 and maximum 16 sequences.
Actual checkpoint dimensions: 512 experts, top-k 10, MoE intermediate 640,
hidden size 2560, 48 layers.

Important ownership distinction: `candidate/model.py` gathers `block_input`
before `self.mlp(block_input)`. HC owns half the rows on each TP2 rank, but the
MoE receives the full prefill batch. Do not estimate MoE rows by dividing the
scheduler budget by TP size.

At an 8192-row prefill, average routed rows per expert are
8192 * 10 / 512 = 160. At 16384 rows they are 320. In the installed b12x
`_select_dynamic_tile_mn` generic gated-NVFP4 policy, both exceed the 96-row
threshold for M128. The W4A8 branch likewise has both above its M64 threshold.
Thus doubling the batch does not cross either policy's large-prefill tile
threshold. Actual prepared tactics can override the generic policy; this is
not a claim about the chosen compiled kernel in every layer.

A 16K scheduler-budget experiment remains technically possible without reducing
precision or maximum context. It would instead test fewer scheduling/collective
boundaries and different larger-shape tuning. It would increase activation and
scratch memory, and could increase decode latency during simultaneous prefill.
It is not a demonstrated improvement and must not be recommended as one.
The current 119-GiB guard and full-context/mixed-load checks would still apply.

Conclusion: deprioritize a speculative 8K-to-16K batch change while conserving
weekly usage. There is no newly discovered tile-threshold shortcut. Keep the
validated 8192-token budget. Evidence comes from the running Docker command,
checkpoint config, HC gather-before-MLP source and installed b12x tile policy.
