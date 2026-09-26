#!/usr/bin/env python3
"""Run the unmodified official harness and retain acceptance/memory evidence."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("tag")
parser.add_argument("--launch-tag")
parser.add_argument("--cases", nargs="+", choices=["prose", "coding", "matrix", "odd", "quick", "c16"], default=["matrix"])
args = parser.parse_args()
root = Path.home() / "builds/qwen-tp2/evidence"
base = "http://127.0.0.1:8000"


def fetch(path):
    with urllib.request.urlopen(base + path, timeout=15) as response:
        return response.read().decode()


for _ in range(120):
    try:
        fetch("/health")
        break
    except Exception:
        launch_path = root / ((args.launch_tag or args.tag) + "-launch.json")
        if launch_path.exists():
            launch = json.loads(launch_path.read_text())
            if not Path(f"/proc/{launch['launcher_pid']}").exists():
                raise RuntimeError("Trial launcher exited before API readiness")
        time.sleep(5)
else:
    raise RuntimeError("Server did not become ready")

# Require both TP2 containers and record their identities before measuring.
for rank in (0, 1):
    command = ["docker", "inspect", "qwen-tp2"]
    if rank: command = ["ssh", "-o", "BatchMode=yes", "10.100.168.1", *command]
    info = json.loads(subprocess.check_output(command))[0]
    assert info["State"]["Running"], f"Rank {rank} is not running"
    cmd = info["Config"]["Cmd"]
    assert cmd[cmd.index("--tensor-parallel-size")+1] == "2"
    (root / f"{args.tag}-rank{rank}-inspect.json").write_text(json.dumps(info, indent=2))

common = [str(Path.home() / "builds/qwen-tp2/.venv/bin/python"), "-u",
          str(Path.home() / "builds/qwen-tp2/llm-inference-bench/llm_decode_bench.py"),
          "--host", "127.0.0.1", "--port", "8000", "--model", "qwen3.8-flash-next-4p89bpw",
          "--no-hw-monitor", "--display-mode", "plain", "--no-resume"]
prose = "Write a detailed step-by-step explanation of how a hash map works, including collision handling, resizing, and time complexity. Be thorough."
cases = {
    "matrix": ["--contexts", "8192,32768,65536", "--concurrency", "1,8",
               "--duration", "30", "--max-tokens", "2048",
               "--standalone-prefill", "--prefill-contexts", "8k,32k,64k"],
    # Concurrencies whose MTP3 verify batches (12/20 tokens) fall between default graph sizes.
    # Screening: one prefill pair and C1/C8 decode at 8K; not a replacement for the full matrix.
    "quick": ["--contexts", "8192", "--concurrency", "1,8", "--duration", "20", "--max-tokens", "2048",
              "--standalone-prefill", "--prefill-contexts", "8k,64k"],
    # Full-occupancy decode: every sequence slot busy (max_num_seqs=16).
    "c16": ["--skip-prefill", "--contexts", "8192,32768", "--concurrency", "16",
            "--duration", "30", "--max-tokens", "2048"],
    "odd": ["--skip-prefill", "--contexts", "8192", "--concurrency", "3,5",
            "--duration", "30", "--max-tokens", "2048"],
    "prose": ["--completion-stats", "--prompt", prose, "--profile-concurrency", "1",
              "--profile-runs", "5", "--max-tokens", "600", "--completion-stats-temperature", "0",
              "--completion-stats-top-p", "1", "--reasoning-effort", "none",
              "--completion-stats-seed", "42", "--completion-stats-correct-regex", "",
              "--completion-stats-no-prefill-scout"],
    "coding": ["--skip-prefill", "--contexts", "8192", "--concurrency", "1",
               "--duration", "15", "--max-tokens", "2048", "--coding-peak",
               "--coding-peak-runs", "3", "--coding-peak-max-tokens", "2000"],
}
for case, options in cases.items():
    if case not in args.cases:
        continue
    prefix = root / f"{args.tag}-{case}"
    command = common + options + ["--output", str(prefix) + ".json"]
    Path(str(prefix) + "-command.json").write_text(json.dumps(command, indent=2))
    Path(str(prefix) + "-metrics-before.txt").write_text(fetch("/metrics"))
    started = time.time()
    with Path(str(prefix) + ".log").open("x") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=900)
    Path(str(prefix) + "-metrics-after.txt").write_text(fetch("/metrics"))
    if result.returncode:
        raise RuntimeError(f"{case} failed: {result.returncode}")
    report = json.loads(Path(str(prefix) + ".json").read_text())
    summary = (report.get("selected_summary") or report.get("coding_peak", {}).get("summary")
               or report.get("summary_table"))
    print(json.dumps({"case": case, "wall_s": time.time() - started, "summary": summary}), flush=True)
