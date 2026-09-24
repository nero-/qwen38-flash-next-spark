#!/usr/bin/env python3
"""Write/check a portable SHA-256 inventory for a downloaded model snapshot."""
import hashlib
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
model = root / "model"
manifest = root / "model.sha256.json"


def inventory() -> dict[str, str]:
    found = {}
    for path in sorted(model.rglob("*")):
        if not path.is_file() or ".cache" in path.relative_to(model).parts:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        found[path.relative_to(model).as_posix()] = digest.hexdigest()
    if not found:
        raise SystemExit(f"No model files found under {model}")
    return found


if sys.argv[1:] == ["write"]:
    manifest.write_text(json.dumps(inventory(), indent=2, sort_keys=True) + "\n")
    print(f"Wrote hashes for {len(json.loads(manifest.read_text()))} model files")
elif sys.argv[1:] == ["check"]:
    expected = json.loads(manifest.read_text())
    actual = inventory()
    missing = sorted(set(expected) - set(actual))
    added = sorted(set(actual) - set(expected))
    changed = sorted(path for path in expected.keys() & actual.keys() if expected[path] != actual[path])
    if missing or added or changed:
        raise SystemExit(f"Snapshot checksum mismatch: missing={missing[:5]}, added={added[:5]}, changed={changed[:5]}")
    print(f"Verified {len(actual)} model file hashes")
else:
    raise SystemExit("Usage: model_manifest.py write|check")
