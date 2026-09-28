#!/usr/bin/env python3
"""Tabulate arms: LIL prefill/decode cells, acceptance, peak RAM and quality summaries.

Steps/s isolates engine speed from MTP acceptance; tok/s = steps/s x accept length
at C1. Read both before crediting a change with a decode gain.
"""
import json, sys
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parent / "evidence"
rows = []
for tag in sys.argv[1:]:
    row = dict(tag=tag)
    for case in ("matrix", "odd", "quick", "c16"):
        path = EVIDENCE / f"{tag}-{case}.json"
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        for ctx, cell in (report.get("prefill") or {}).items():
            row[f"pp{int(ctx) // 1024}k"] = cell["tok_per_sec"]
        for cell in report["results"]:
            key = f"c{cell['concurrency']}@{cell['context_tokens'] // 1024}k"
            assert not cell.get("num_errors") and not cell.get("underfilled"), (tag, key)
            row[key] = round(cell["aggregate_tps"], 1)
            row[key + " steps"] = round(cell["server_steps_per_s"] or 0, 2)
            row[key + " acc"] = round(cell["server_spec_accept_length"] or 0, 3)
    memory = EVIDENCE / f"{tag}-bench-memory.jsonl"
    if memory.exists():
        used = [json.loads(line)["used_bytes"] for line in memory.read_text().splitlines() if line.strip()]
        row["peak_used_gib"] = round(max(used) / 2**30, 2)
    q = EVIDENCE / f"{tag}-qeval.json"
    if q.exists():
        r = json.loads(q.read_text())
        row.update({f"nll_{k}": round(v["mean_nll"], 5) for k, v in r["nll"].items()},
                   gsm8k=r["gsm8k"]["accuracy"])
    m = EVIDENCE / f"{tag}-mmlu.json"
    if m.exists():
        row["mmlu"] = json.loads(m.read_text())["accuracy"]
    rows.append(row)
print(json.dumps(rows, indent=1))
