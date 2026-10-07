#!/usr/bin/env python3
"""Verify the NVFP4-CSF download and build the vLLM serving directory for it.

prepare_csf.py verify   check SHA256SUMS against node-config.json, then every file against it
prepare_csf.py serve    (re)build serve-csf/: metadata/ files, with config.json's
                        quantization_config pointing at model-csf/ as the NVFP4-CSF reader requires
"""
import hashlib
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from config import load_config

ROOT = Path(__file__).resolve().parent
cfg = load_config(ROOT)
checkpoint = ROOT / "model-csf"
serve = ROOT / "serve-csf"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify() -> None:
    sums = checkpoint / "SHA256SUMS"
    if sha256(sums) != cfg["csf_model"]["sha256sums_sha256"]:
        raise SystemExit("SHA256SUMS does not match node-config.json csf_model.sha256sums_sha256")
    expected = {}
    for line in sums.read_text().splitlines():
        digest, name = line.split(None, 1)
        expected[name.strip().removeprefix("./")] = digest
    with ThreadPoolExecutor(8) as pool:
        actual = dict(zip(expected, pool.map(lambda n: sha256(checkpoint / n) if (checkpoint / n).is_file() else None,
                                             expected)))
    bad = sorted(name for name in expected if actual[name] != expected[name])
    if bad:
        raise SystemExit(f"{len(bad)} CSF files missing or corrupt: {bad[:10]}")
    print(f"Verified {len(expected)} CSF files against SHA256SUMS")


def build_serve() -> None:
    metadata = checkpoint / "metadata"
    config = json.loads((metadata / "config.json").read_text())
    source = config["quantization_config"]
    if source.get("quant_method") == "nvfp4_csf":
        raise SystemExit("metadata/config.json is already rewritten; expected the source ModelOpt config")
    config["quantization_config"] = {
        "quant_method": "nvfp4_csf",
        "format_version": 1,
        "checkpoint_root": str(checkpoint.resolve()),
        "source_quantization_config": source,
    }
    staging = serve.with_name(serve.name + ".tmp")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    for item in metadata.iterdir():
        if item.is_file() and item.name != "config.json":
            shutil.copy2(item, staging / item.name)
    (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    shutil.rmtree(serve, ignore_errors=True)
    staging.rename(serve)
    print(f"Built {serve} with checkpoint_root {config['quantization_config']['checkpoint_root']}")


if __name__ == "__main__":
    if sys.argv[1:] == ["verify"]:
        verify()
    elif sys.argv[1:] == ["serve"]:
        build_serve()
    else:
        raise SystemExit(__doc__)
