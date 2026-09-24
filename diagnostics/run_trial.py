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
parser.add_argument("--cases", nargs="+", choices=["prose", "coding", "matrix"], default=["matrix"])
args = parser.parse_args()
root = Path.home() / "bench/tp1-opt-20260922"
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

# A failed trial must not silently benchmark a later restored server.
launch_path = root / ((args.launch_tag or args.tag) + "-launch.json")
if launch_path.exists():
    launch = json.loads(launch_path.read_text())
    if "container_id" in launch:
        info = json.loads(subprocess.check_output(
            ["docker", "inspect", launch["container_id"]]))[0]
        if (not info["State"]["Running"] or info["Image"] != launch["image_id"]
                or info["State"]["Pid"] != launch["launcher_pid"]):
            raise RuntimeError("Running container does not match trial image/process")
    expected = launch["env"]
    verified = False
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            argv = (proc / "cmdline").read_bytes().split(b"\0")
            if b"vllm.entrypoints.cli.main" not in argv:
                continue
            if "container_id" in launch and int(proc.name) != launch["launcher_pid"]:
                continue
            formats = [argv[i + 1].decode() for i, arg in enumerate(argv[:-1])
                       if arg == b"--load-format"]
            if "load_format" in launch and (not formats or formats[-1] != launch["load_format"]):
                raise RuntimeError("Running server does not match trial loader")
            entries = (proc / "environ").read_bytes().split(b"\0")
            actual = dict(e.decode().split("=", 1) for e in entries if b"=" in e)
            if any(actual.get(k) != v for k, v in expected.items()):
                raise RuntimeError("Running server does not match trial environment")
            verified = True
        except (FileNotFoundError, PermissionError):
            pass
    if not verified:
        raise RuntimeError("Could not verify trial server identity")

common = [str(Path.home() / "venvs/qwen38/bin/python"), "-u",
          str(Path.home() / "projects/llm-inference-bench/llm_decode_bench.py"),
          "--host", "127.0.0.1", "--port", "8000", "--model", "qwen3.8-flash-next-4p89bpw",
          "--no-hw-monitor", "--display-mode", "plain", "--no-resume"]
prose = "Write a detailed step-by-step explanation of how a hash map works, including collision handling, resizing, and time complexity. Be thorough."
cases = {
    "matrix": ["--contexts", "8192,32768,65536", "--concurrency", "1,8",
               "--duration", "30", "--max-tokens", "2048",
               "--standalone-prefill", "--prefill-contexts", "8k,32k,64k"],
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
