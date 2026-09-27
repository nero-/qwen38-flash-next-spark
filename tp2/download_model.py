#!/usr/bin/env python3
"""Download an immutable model snapshot with the pinned runtime image.

download_model.py [main|hybrid|trunk]: main -> model/ (config "model"),
hybrid -> model-5500h/ (config "hybrid_model", the default serving checkpoint),
trunk -> model-5500/ (config "trunk_model", only for building the hybrid from source).
"""
import os
import sys
from pathlib import Path

from config import load_config

ROOT = Path(__file__).resolve().parent
cfg = load_config(ROOT)
which = sys.argv[1] if len(sys.argv) > 1 else "main"
key, directory = {"main": ("model", "model"), "hybrid": ("hybrid_model", "model-5500h"),
                  "trunk": ("trunk_model", "model-5500")}[which]
target = Path.home() / cfg["remote_root"] / directory
target.mkdir(parents=True, exist_ok=True)
# The Xet transfer backend needs a writable cache; keep it inside the (excluded) snapshot .cache.
os.environ.setdefault("HF_HOME", str(target / ".cache/hf"))
from huggingface_hub import snapshot_download  # noqa: E402  (after HF_HOME)

snapshot_download(
    repo_id=cfg[key]["repository"],
    revision=cfg[key]["revision"],
    local_dir=str(target),
    token=os.environ.get("HF_TOKEN") or None,
    max_workers=8,
)
if not (target / "config.json").is_file() or not (target / "model.safetensors.index.json").is_file():
    raise SystemExit(f"Snapshot is incomplete: {target}")
print(f"Downloaded {cfg[key]['repository']}@{cfg[key]['revision']} to {target}")
