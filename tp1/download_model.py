#!/usr/bin/env python3
"""Download the pinned step-5500 hybrid checkpoint into model-5500h/ (run inside the pinned image)."""
import os
from pathlib import Path

from config import load_config

ROOT = Path(__file__).resolve().parent
cfg = load_config(ROOT)
target = ROOT / "model-5500h"
target.mkdir(parents=True, exist_ok=True)
# The Xet transfer backend needs a writable cache; keep it inside the snapshot's ignored .cache.
os.environ.setdefault("HF_HOME", str(target / ".cache/hf"))
from huggingface_hub import snapshot_download  # noqa: E402  (after HF_HOME)

snapshot_download(
    repo_id=cfg["hybrid_model"]["repository"],
    revision=cfg["hybrid_model"]["revision"],
    local_dir=str(target),
    token=os.environ.get("HF_TOKEN") or None,
    max_workers=8,
)
if not (target / "config.json").is_file() or not (target / "model.safetensors.index.json").is_file():
    raise SystemExit(f"Snapshot is incomplete: {target}")
print(f"Downloaded {cfg['hybrid_model']['repository']}@{cfg['hybrid_model']['revision']} to {target}")
