#!/usr/bin/env python3
"""Safely select one or two DAC cables for both TP2 ranks."""
import argparse
import datetime
import subprocess
import urllib.request
from pathlib import Path

from config import active_requests, load_config, peer_command

HERE = Path(__file__).resolve().parent
cfg = load_config(HERE)
root = Path.home() / cfg["remote_root"]
worker = cfg["ranks"][1]["peer_ssh"]
peer = ["ssh", *cfg["ssh_options"], worker]
p = argparse.ArgumentParser()
p.add_argument("cables", type=int, choices=[1, 2])
requested = p.parse_args().cables
selected = root / "selected-cables.txt"
old = int(selected.read_text().strip()) if selected.exists() else cfg["default_cables"]
remote_selected = subprocess.run(peer_command(cfg, ["cat", f"$HOME/{cfg['remote_root']}/selected-cables.txt"]), capture_output=True, text=True)
if remote_selected.returncode != 0 or remote_selected.stdout.strip() != str(old):
    raise SystemExit("Rank cable selections differ or are missing; repair both selected-cables.txt files before switching")
if requested == old:
    print(f"Both ranks already use {old} cable mode")
    raise SystemExit(0)
profile = (root / "selected-profile.txt").read_text().strip() if (root / "selected-profile.txt").exists() else cfg["default_profile"]
for rank in (0, 1):
    script = f"$HOME/{cfg['remote_root']}/launch.py" if rank else str(root / "launch.py")
    args = ["python3", script, "check", "--rank", str(rank), "--profile", profile, "--cables", str(requested)]
    if rank:
        subprocess.run(peer_command(cfg, args), check=True, stdout=subprocess.DEVNULL)
    else:
        subprocess.run(args, check=True, stdout=subprocess.DEVNULL)
try:
    metrics = urllib.request.urlopen(f"http://127.0.0.1:{cfg['api_port']}/metrics", timeout=5).read().decode()
except OSError:
    metrics = ""
    local_running = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", "qwen-tp2"], capture_output=True, text=True).stdout.strip()
    remote_running = subprocess.run(peer_command(cfg, ["docker", "inspect", "-f", "{{.State.Running}}", "qwen-tp2"]), capture_output=True, text=True).stdout.strip()
    if local_running == "true" or remote_running == "true":
        raise SystemExit("Cannot establish that the running service is idle; metrics are unavailable")
if not any(line.startswith("vllm:num_requests_") for line in metrics.splitlines()):
    local_running = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", "qwen-tp2"], capture_output=True, text=True).stdout.strip()
    remote_running = subprocess.run(peer_command(cfg, ["docker", "inspect", "-f", "{{.State.Running}}", "qwen-tp2"]), capture_output=True, text=True).stdout.strip()
    if local_running == "true" or remote_running == "true":
        raise SystemExit("Cannot establish that the running service is idle; request metrics are missing")
if active_requests(metrics):
    raise SystemExit("Active requests: wait before switching cable mode")

stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def retain(rank: int) -> None:
    prefix = peer if rank else []
    result = subprocess.run(prefix + ["docker", "inspect", "qwen-tp2"], capture_output=True, text=True)
    if result.returncode == 0:
        (root / "evidence").mkdir(exist_ok=True)
        (root / "evidence" / f"{stamp}-cables-old-rank{rank}.json").write_text(result.stdout)
        with (root / "evidence" / f"{stamp}-cables-old-rank{rank}.log").open("w") as log:
            subprocess.run(prefix + ["docker", "logs", "qwen-tp2"], stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(prefix + ["docker", "rm", "-f", "qwen-tp2"], check=True)


def write_mode(value: int) -> None:
    selected.write_text(f"{value}\n")
    subprocess.run(["rsync", "-a", str(selected), f"{worker}:{cfg['remote_root']}/selected-cables.txt"], check=True)


try:
    subprocess.run(["bash", str(root / "cluster.sh"), "stop"], check=False)
    for rank in (0, 1):
        retain(rank)
    write_mode(requested)
    subprocess.run(["bash", str(root / "cluster.sh"), "start"], check=True)
    subprocess.run(["bash", str(root / "cluster.sh"), "wait"], check=True)
except Exception:
    subprocess.run(["bash", str(root / "cluster.sh"), "stop"], check=False)
    for rank in (0, 1):
        prefix = peer if rank else []
        subprocess.run(prefix + ["docker", "rm", "-f", "qwen-tp2"], check=False, stdout=subprocess.DEVNULL)
    write_mode(old)
    subprocess.run(["bash", str(root / "cluster.sh"), "start"], check=False)
    raise
