#!/usr/bin/env python3
"""Run the unmodified official harness against qwen-tp1; keep metrics and RAM evidence.

bench.py TAG [--cases matrix quick odd c16 coding prose]
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("tag")
parser.add_argument("--cases", nargs="+", choices=["prose", "coding", "matrix", "odd", "quick", "c16"], default=["matrix"])
args = parser.parse_args()
ROOT = Path(__file__).resolve().parent
root = ROOT / "evidence"
root.mkdir(exist_ok=True)
base = "http://127.0.0.1:8000"


def fetch(path):
    with urllib.request.urlopen(base + path, timeout=15) as response:
        return response.read().decode()


for _ in range(300):
    try:
        fetch("/health")
        break
    except Exception:
        time.sleep(5)
else:
    raise RuntimeError("Server did not become ready")

info = json.loads(subprocess.check_output(["docker", "inspect", "qwen-tp1"]))[0]
assert info["State"]["Running"], "qwen-tp1 is not running"
(root / f"{args.tag}-inspect.json").write_text(json.dumps(info, indent=2))

common = [str(ROOT / ".venv/bin/python"), "-u", str(ROOT / "llm-inference-bench/llm_decode_bench.py"),
          "--host", "127.0.0.1", "--port", "8000", "--model", "qwen3.8-flash-next-4p89bpw",
          "--no-hw-monitor", "--display-mode", "plain", "--no-resume"]
prose = "Write a detailed step-by-step explanation of how a hash map works, including collision handling, resizing, and time complexity. Be thorough."
cases = {
    "matrix": ["--contexts", "8192,32768,65536", "--concurrency", "1,8",
               "--duration", "30", "--max-tokens", "2048",
               "--standalone-prefill", "--prefill-contexts", "8k,32k,64k"],
    # Screening: prefill at 8K/64K and C1/C8 decode at 8K; not a replacement for the full matrix.
    "quick": ["--contexts", "8192", "--concurrency", "1,8", "--duration", "20", "--max-tokens", "2048",
              "--standalone-prefill", "--prefill-contexts", "8k,64k"],
    # Full-occupancy decode: every sequence slot busy (max_num_seqs=16).
    "c16": ["--skip-prefill", "--contexts", "8192,32768", "--concurrency", "16",
            "--duration", "30", "--max-tokens", "2048"],
    # Concurrencies whose MTP3 verify batches (12/20 tokens) fall between default graph sizes.
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
watch = subprocess.Popen(["python3", str(ROOT / "watch_memory.py"), "--tag", f"{args.tag}-bench", "--seconds", "7200"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for case, options in cases.items():
        if case not in args.cases:
            continue
        prefix = root / f"{args.tag}-{case}"
        command = common + options + ["--output", str(prefix) + ".json"]
        Path(str(prefix) + "-command.json").write_text(json.dumps(command, indent=2))
        Path(str(prefix) + "-metrics-before.txt").write_text(fetch("/metrics"))
        started = time.time()
        with Path(str(prefix) + ".log").open("x") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        Path(str(prefix) + "-metrics-after.txt").write_text(fetch("/metrics"))
        if result.returncode:
            raise RuntimeError(f"{case} failed: {result.returncode}")
        report = json.loads(Path(str(prefix) + ".json").read_text())
        summary = (report.get("selected_summary") or report.get("coding_peak", {}).get("summary")
                   or report.get("summary_table"))
        print(json.dumps({"case": case, "wall_s": round(time.time() - started), "summary": summary}), flush=True)
finally:
    watch.terminate()
rows = (root / f"{args.tag}-bench-memory.jsonl").read_text().splitlines()
peak = max(json.loads(line)["used_bytes"] for line in rows if line.strip()) / 2**30
print(json.dumps({"tag": args.tag, "peak_used_gib": round(peak, 2)}), flush=True)
