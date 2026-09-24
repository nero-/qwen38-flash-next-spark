#!/usr/bin/env python3
"""Download the immutable model snapshot with the pinned runtime image."""
import os
from pathlib import Path

from huggingface_hub import snapshot_download

from config import load_config

ROOT = Path(__file__).resolve().parent
cfg = load_config(ROOT)
target = Path.home() / cfg["remote_root"] / "model"
target.mkdir(parents=True, exist_ok=True)
snapshot_download(
    repo_id=cfg["model"]["repository"],
    revision=cfg["model"]["revision"],
    local_dir=str(target),
    token=os.environ.get("HF_TOKEN") or None,
    max_workers=6,
)
if not (target / "config.json").is_file() or not (target / "model.safetensors.index.json").is_file():
    raise SystemExit(f"Snapshot is incomplete: {target}")
print(f"Downloaded {cfg['model']['repository']}@{cfg['model']['revision']} to {target}")
