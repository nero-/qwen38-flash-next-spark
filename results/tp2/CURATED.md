# Curated TP2 receipts

These tracked JSON records support the documented profile, long-context, and cable comparisons. They exclude machine-specific Docker inspections, commands containing local paths, full metrics dumps, per-second memory streams, server logs, and profiler traces. The excluded local files are ignored by Git and were left untouched on disk.

The HC/adaptive matrix and repeat/control runs remain separate so the overlapping ranges and uncertain performance conclusion can be reviewed directly. Cable files retain the final matched one/two-cable model matrices and generated communication/model summaries. This evidence is from the original cluster only; it is not a clean-install qualification.

`campaign-20260926/` holds the September 26 arms: LIL matrix, odd-concurrency, quick,
C16, determinism and capacity JSON plus a generated `summary.json` (speed, quality
summaries, peak RAM, KV capacity, failed-arm causes). Per-token quality records and
per-question answers stay on rank 0 under `evidence/` because of their size.
