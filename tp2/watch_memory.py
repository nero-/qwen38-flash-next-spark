#!/usr/bin/env python3
"""Record RAM headroom; guard a used-memory ceiling or 3 GiB available floor."""
import argparse
import json
import os
from pathlib import Path
import signal
import time

parser = argparse.ArgumentParser()
parser.add_argument("--tag", required=True)
parser.add_argument("--seconds", type=int, default=900)
parser.add_argument("--guard", action="store_true")
parser.add_argument("--max-used-gib", type=float)
args = parser.parse_args()
root = Path.home() / "builds/qwen-tp2/evidence"
root.mkdir(exist_ok=True)
low = 0
with (root / (args.tag + "-memory.jsonl")).open("x") as out:
    for _ in range(args.seconds):
        fields = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in ("MemTotal", "MemAvailable", "MemFree", "SwapFree", "SwapTotal"):
                fields[key] = int(value.split()[0]) * 1024
        fields["time"] = time.time()
        fields["used_bytes"] = fields["MemTotal"] - fields["MemAvailable"]
        exceeded = (fields["used_bytes"] > args.max_used_gib * 2**30
                    if args.max_used_gib is not None else fields["MemAvailable"] < 3 * 2**30)
        low = low + 1 if exceeded else 0
        if args.guard and low >= 3:
            killed = []
            for proc in Path("/proc").glob("[0-9]*"):
                try:
                    argv = (proc / "cmdline").read_bytes().split(b"\0")
                    if b"vllm.entrypoints.cli.main" in argv and str(Path.home() / "builds/qwen-tp2/model").encode() in argv:
                        pid = int(proc.name)
                        os.kill(pid, signal.SIGTERM)
                        killed.append(pid)
                except (OSError, ValueError):
                    pass
            fields["guard_stopped_server_pids"] = killed
            out.write(json.dumps(fields) + "\n")
            out.flush()
            print(json.dumps(fields), flush=True)
            raise SystemExit(2)
        out.write(json.dumps(fields) + "\n")
        out.flush()
        time.sleep(1)
