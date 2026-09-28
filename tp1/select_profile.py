#!/usr/bin/env python3
"""Switch the idle single-Spark server to a profile, rolling back on failure."""
import argparse
import datetime
import subprocess
import urllib.request
from pathlib import Path

from config import active_requests, load_config, parse_profile

cfg = load_config(Path(__file__).resolve().parent)
root = Path.home() / cfg["remote_root"]
p = argparse.ArgumentParser()
p.add_argument("profile")
requested = p.parse_args().profile
try:
    parse_profile(requested)
except ValueError as exc:
    p.error(str(exc))
selection = root / "selected-profile.txt"
old = selection.read_text().strip() if selection.exists() else cfg["default_profile"]


def running() -> bool:
    out = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", "qwen-tp1"], capture_output=True, text=True)
    return out.stdout.strip() == "true"


def ensure_idle() -> None:
    try:
        metrics = urllib.request.urlopen(f"http://127.0.0.1:{cfg['api_port']}/metrics", timeout=5).read().decode()
    except OSError:
        metrics = ""
    if not any(line.startswith("vllm:num_requests_") for line in metrics.splitlines()):
        if running():
            raise SystemExit("Cannot establish that the running server is idle; request metrics are unavailable")
        return
    if active_requests(metrics):
        raise SystemExit("Active requests: wait before switching profiles")


def retire(stamp: str, label: str) -> None:
    inspect = subprocess.run(["docker", "inspect", "qwen-tp1"], capture_output=True, text=True)
    if inspect.returncode:
        return
    evidence = root / "evidence"
    evidence.mkdir(exist_ok=True)
    (evidence / f"{stamp}-{label}.json").write_text(inspect.stdout)
    with (evidence / f"{stamp}-{label}.log").open("w") as log:
        subprocess.run(["docker", "logs", "qwen-tp1"], stdout=log, stderr=subprocess.STDOUT, check=False)
    subprocess.run(["docker", "stop", "--time", "60", "qwen-tp1"], check=False, stdout=subprocess.DEVNULL)
    subprocess.run(["docker", "rm", "qwen-tp1"], check=True, stdout=subprocess.DEVNULL)


subprocess.run(["python3", str(root / "launch.py"), "check", "--profile", requested], check=True, stdout=subprocess.DEVNULL)
ensure_idle()
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
retire(stamp, "retired")
selection.write_text(requested + "\n")
try:
    subprocess.run(["bash", str(root / "node.sh"), "start"], check=True)
    subprocess.run(["bash", str(root / "node.sh"), "wait"], check=True)
except Exception:
    # Keep the failed candidate's logs; they are the only record of why it failed.
    retire(stamp, f"failed-{requested.replace('+', '_')}")
    selection.write_text(old + "\n")
    subprocess.run(["bash", str(root / "node.sh"), "start"], check=False)
    raise
print(f"Serving {requested}")
