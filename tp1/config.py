"""Single-Spark (TP1) node configuration and profile parsing."""
from __future__ import annotations

import json
import re
from pathlib import Path

# Each modifier changes exactly one axis of the base "tp1" profile.
FLAG_MODIFIERS = {
    "cg4",         # capture every MTP verify size exactly instead of padding to default sizes
    "mxhead",      # MXFP8 target LM head weights (base keeps BF16 head weights)
    "greedyspec",  # vLLM default draft/rejection sampling instead of probabilistic/block
    "fp32ssm",     # checkpoint-native FP32 recurrent state (base: BF16)
    "resident",    # PLE table resident in memory (base: read rows from the checkpoint on disk)
}
PARAM_MODIFIERS = {
    "kv": (1, 40),     # kvN: N GiB of FP8 KV cache, overriding kv_cache_gib
    "mtp": (1, 6),     # mtpN: N speculative tokens (base: 3)
    "seqs": (1, 32),   # seqsN: max concurrent sequences (base: 16)
}
PROFILE_RE = re.compile(r"tp1(?:\+[a-z0-9]+)*")


def load_config(root: Path) -> dict:
    data = json.loads((root / "node-config.json").read_text())
    validate_config(data)
    return data


def validate_config(data: dict) -> None:
    if not re.fullmatch(r"(?:[A-Za-z0-9_.-]+@)?[A-Za-z0-9_.:-]+", data["ssh"]):
        raise ValueError(f"unsafe SSH destination: {data['ssh']}")
    if not re.fullmatch(r"(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+", data["remote_root"]):
        raise ValueError("remote_root must be a relative path with safe shell characters")
    if not 1 <= int(data["api_port"]) <= 65535:
        raise ValueError("api_port must be in 1..65535")
    if not re.fullmatch(r"[A-Za-z0-9_./-]+@sha256:[0-9a-f]{64}", data["image"]):
        raise ValueError("image must be pinned by a SHA-256 OCI digest")
    model = data["hybrid_model"]
    if not re.fullmatch(r"[0-9a-f]{40}", model["revision"]):
        raise ValueError("hybrid_model.revision must be a full immutable 40-character commit")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model["repository"]):
        raise ValueError("hybrid_model.repository must be an owner/name identifier")
    if not 1 <= int(data["kv_cache_gib"]) <= 40:
        raise ValueError("kv_cache_gib must be a whole number of GiB in 1..40")
    parse_profile(data["default_profile"])


def parse_profile(profile: str) -> dict:
    """Return {"flags": set, "kv": int|None, "mtp": int|None, "seqs": int|None}."""
    if not PROFILE_RE.fullmatch(profile):
        raise ValueError(f"profile must look like tp1[+modifier...]: {profile}")
    parsed = {"flags": set(), **{key: None for key in PARAM_MODIFIERS}}
    modifiers = profile.split("+")[1:]
    if len(set(modifiers)) != len(modifiers):
        raise ValueError(f"repeated modifier in {profile}")
    for mod in modifiers:
        if mod in FLAG_MODIFIERS:
            parsed["flags"].add(mod)
            continue
        match = re.fullmatch(r"([a-z]+)(\d+)", mod)
        if not match or match.group(1) not in PARAM_MODIFIERS:
            raise ValueError(f"unknown modifier {mod!r}; known: {sorted(FLAG_MODIFIERS)} "
                             f"and {sorted(k + 'N' for k in PARAM_MODIFIERS)}")
        key, value = match.group(1), int(match.group(2))
        low, high = PARAM_MODIFIERS[key]
        if parsed[key] is not None or not low <= value <= high:
            raise ValueError(f"{key} must be given once, in {low}..{high}")
        parsed[key] = value
    return parsed


def active_requests(metrics: str) -> bool:
    for line in metrics.splitlines():
        if not line.startswith(("vllm:num_requests_running{", "vllm:num_requests_waiting{")):
            continue
        try:
            if float(line.split()[-1]) > 0:
                return True
        except (ValueError, IndexError):
            raise ValueError(f"Malformed request metric: {line}")
    return False


if __name__ == "__main__":
    import sys

    value = load_config(Path(__file__).resolve().parent)
    for key in sys.argv[1].split("."):
        value = value[key]
    if isinstance(value, (dict, list)):
        raise SystemExit("config lookup must select a scalar value")
    print(value)
