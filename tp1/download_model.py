#!/usr/bin/env python3
"""Download a pinned checkpoint (run inside the pinned image).

download_model.py [hybrid|csf]: hybrid -> model-5500h/ (config "hybrid_model"),
csf -> model-csf/ (config "csf_model", the NVFP4-CSF container of the same weights).
"""
import os
import sys
from pathlib import Path

from config import load_config

ROOT = Path(__file__).resolve().parent
cfg = load_config(ROOT)
which = sys.argv[1] if len(sys.argv) > 1 else "hybrid"
key, directory, marker = {"hybrid": ("hybrid_model", "model-5500h", "model.safetensors.index.json"),
                          "csf": ("csf_model", "model-csf", "manifest.json")}[which]
target = ROOT / directory
target.mkdir(parents=True, exist_ok=True)
# The Xet transfer backend needs a writable cache; keep it inside the snapshot's ignored .cache.
os.environ.setdefault("HF_HOME", str(target / ".cache/hf"))
from huggingface_hub import snapshot_download  # noqa: E402  (after HF_HOME)

snapshot_download(
    repo_id=cfg[key]["repository"],
    revision=cfg[key]["revision"],
    local_dir=str(target),
    token=os.environ.get("HF_TOKEN") or None,
    max_workers=8,
)
if not (target / "config.json").is_file() or not (target / marker).is_file():
    raise SystemExit(f"Snapshot is incomplete: {target}")
print(f"Downloaded {cfg[key]['repository']}@{cfg[key]['revision']} to {target}")
