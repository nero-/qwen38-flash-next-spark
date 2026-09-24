"""Extract the official LIL matrix without replacing its measurement definitions."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("tags", nargs="+")
args = parser.parse_args()
root = Path.home() / "bench/tp1-opt-20260922"
reports = {}
for tag in args.tags:
    path = root / f"{tag}-matrix.json"
    report = json.loads(path.read_text())
    cells = []
    for cell in report["results"]:
        cells.append({key: cell.get(key) for key in (
            "context_tokens", "concurrency", "aggregate_tps", "per_request_avg_tps",
            "input_seq_len_avg", "measurement_seconds", "num_errors",
            "failure_reason", "server_steps_per_s", "server_spec_accept_length",
            "effective_concurrency", "underfilled", "warmup_timed_out",
            "capacity_limited", "loop_detected")})
    reports[tag] = {"metadata": report["metadata"], "prefill": report["prefill"],
                    "cells": cells}
(root / "lil-matrix-summary.json").write_text(json.dumps(reports, indent=2))
print(json.dumps(reports, indent=2))
