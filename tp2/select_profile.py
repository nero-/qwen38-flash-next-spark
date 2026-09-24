#!/usr/bin/env python3
"""Switch both idle TP2 ranks to a validated profile with rollback on failure."""
import argparse
import datetime
import json
import subprocess
import urllib.request
from pathlib import Path

from config import active_requests, load_config, peer_command

ROOT = Path(__file__).resolve().parent
cfg = load_config(ROOT)
root = Path.home() / cfg["remote_root"]
worker = cfg["ranks"][1]["peer_ssh"]
peer = ["ssh", *cfg["ssh_options"], worker]
p = argparse.ArgumentParser()
p.add_argument("profile", choices=["baseline", "hc", "hc-adaptive", "hc-prefill", "hc-prefill-dynamic", "hc-k20-mtp5"])
requested = p.parse_args().profile
old = (root / "selected-profile.txt").read_text().strip() if (root / "selected-profile.txt").exists() else cfg["default_profile"]


def run(rank: int, action: str, *, check: bool = True):
    local = ["python3", str(root / "launch.py"), action, "--rank", str(rank)]
    if rank:
        return subprocess.run(peer_command(cfg, local), check=check, text=True, capture_output=not check)
    return subprocess.run(local, check=check, text=True, capture_output=not check)


def save_receipt(rank: int, stamp: str) -> None:
    prefix = peer if rank else []
    inspect = subprocess.run(prefix + ["docker", "inspect", "qwen-tp2"], capture_output=True, text=True)
    if inspect.returncode == 0:
        (root / "evidence").mkdir(exist_ok=True)
        (root / "evidence" / f"{stamp}-retired-rank{rank}.json").write_text(inspect.stdout)
        with (root / "evidence" / f"{stamp}-retired-rank{rank}.log").open("w") as log:
            subprocess.run(prefix + ["docker", "logs", "qwen-tp2"], stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(prefix + ["docker", "stop", "--timeout", "60", "qwen-tp2"], check=True)
        subprocess.run(prefix + ["docker", "rm", "qwen-tp2"], check=True)


def write_selection(name: str) -> None:
    (root / "selected-profile.txt").write_text(name + "\n")
    subprocess.run(["rsync", "-a", str(root / "selected-profile.txt"), f"{worker}:{cfg['remote_root']}/selected-profile.txt"], check=True)


def switch(name: str) -> None:
    for rank in (0, 1):
        script = f"$HOME/{cfg['remote_root']}/launch.py" if rank else str(root / "launch.py")
        args = ["python3", script, "check", "--rank", str(rank), "--profile", name]
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
    has_request_metrics = any(line.startswith("vllm:num_requests_") for line in metrics.splitlines())
    if not has_request_metrics:
        local_running = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", "qwen-tp2"], capture_output=True, text=True).stdout.strip()
        remote_running = subprocess.run(peer_command(cfg, ["docker", "inspect", "-f", "{{.State.Running}}", "qwen-tp2"]), capture_output=True, text=True).stdout.strip()
        if local_running == "true" or remote_running == "true":
            raise SystemExit("Cannot establish that the running service is idle; request metrics are missing")
    if active_requests(metrics):
        raise SystemExit("Active requests: wait before switching profiles")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        for rank in (0, 1):
            save_receipt(rank, stamp)
        write_selection(name)
        subprocess.run(["bash", str(root / "cluster.sh"), "start"], check=True)
        subprocess.run(["bash", str(root / "cluster.sh"), "wait"], check=True)
    except Exception:
        subprocess.run(["bash", str(root / "cluster.sh"), "stop"], check=False)
        for rank in (0, 1):
            prefix = peer if rank else []
            subprocess.run(prefix + ["docker", "rm", "-f", "qwen-tp2"], check=False, stdout=subprocess.DEVNULL)
        write_selection(old)
        subprocess.run(["bash", str(root / "cluster.sh"), "start"], check=False)
        raise


switch(requested)
