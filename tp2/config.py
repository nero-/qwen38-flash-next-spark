"""Shared, portable TP2 cluster configuration and RoCE rail discovery."""
from __future__ import annotations

import ipaddress
import json
import re
import shlex
from pathlib import Path


def load_config(root: Path) -> dict:
    path = root / "cluster-config.json"
    data = json.loads(path.read_text())
    validate_config(data)
    return data


def validate_config(data: dict) -> None:
    if len(data.get("ranks", [])) != 2:
        raise ValueError("TP2 config must define exactly two ranks")
    if data.get("default_cables") not in (1, 2):
        raise ValueError("default_cables must be 1 or 2")
    if not re.fullmatch(r"(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+", data["remote_root"]):
        raise ValueError("remote_root must be a relative path with safe shell characters")
    if not (1 <= int(data["api_port"]) <= 65535 and 1 <= int(data["master_port"]) <= 65535):
        raise ValueError("API and master ports must be in 1..65535")
    for key in ("model", "trunk_model"):
        if not re.fullmatch(r"[0-9a-f]{40}", data[key]["revision"]):
            raise ValueError(f"{key}.revision must be a full immutable 40-character commit")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", data[key]["repository"]):
            raise ValueError(f"{key}.repository must be an owner/name identifier")
    if not 1 <= int(data["kv_cache_gib"]) <= 64:
        raise ValueError("kv_cache_gib must be a whole number of GiB in 1..64")
    if not re.fullmatch(r"[A-Za-z0-9_./-]+@sha256:[0-9a-f]{64}", data["image"]):
        raise ValueError("image must be pinned by a SHA-256 OCI digest")
    for key, image in data.get("experimental_images", {}).items():
        # Local derived tags are allowed here; launch receipts record their image IDs.
        if not re.fullmatch(r"[a-z0-9]+", key) or not re.fullmatch(r"[A-Za-z0-9_./:-]+(?:@sha256:[0-9a-f]{64})?", image):
            raise ValueError(f"unsafe experimental image entry: {key}")
    allowed_profiles = {"baseline", "hc", "hc-adaptive", "hc-prefill", "hc-prefill-dynamic", "hc-k20-mtp5"}
    # Modifiers after "+" are validated by launch.py check before any start.
    if not re.fullmatch(r"[a-z0-9-]+(?:\+[a-z0-9]+)*", data.get("default_profile", "")) \
            or data["default_profile"].split("+")[0] not in allowed_profiles:
        raise ValueError("default_profile is not an installable profile")
    for rank in data["ranks"]:
        ipaddress.ip_address(rank["ip"])
        for key in ("ssh", "peer_ssh"):
            if not re.fullmatch(r"(?:[A-Za-z0-9_.-]+@)?[A-Za-z0-9_.:-]+", rank[key]):
                raise ValueError(f"unsafe SSH destination: {rank[key]}")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", rank["socket_interface"]):
            raise ValueError("unsafe socket interface name")
        if set(rank["cables"]) != {"1", "2"}:
            raise ValueError("each rank must map both cable modes")
        for mode, rails in rank["cables"].items():
            if len(rails) != 2:
                raise ValueError(f"cable mode {mode} must select two HCA rails")
            for rail in rails:
                if not all(rail.get(key) for key in ("interface", "hca")):
                    raise ValueError("each rail needs interface and hca")
                if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", rail[key]) for key in ("interface", "hca")):
                    raise ValueError("unsafe interface or HCA name")


def selected_rails(config: dict, rank: int, cables: int) -> list[dict]:
    if rank not in (0, 1) or cables not in (1, 2):
        raise ValueError("rank must be 0 or 1 and cables must be 1 or 2")
    return config["ranks"][rank]["cables"][str(cables)]


def peer_command(config: dict, args: list[str]) -> list[str]:
    """Build a single remote SSH command while preserving argv boundaries."""
    quoted = [f'"{arg}"' if arg.startswith("$HOME/") else shlex.quote(arg) for arg in args]
    return ["ssh", *config["ssh_options"], config["ranks"][1]["peer_ssh"], " ".join(quoted)]


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


def find_ipv4_gid(hca_path: Path, interface: str) -> int:
    """Find an IPv4 RoCE GID whose recorded netdev is the selected DAC NIC."""
    port = hca_path / "ports/1"
    matches = []
    for ndev in sorted((port / "gid_attrs/ndevs").iterdir(), key=lambda p: int(p.name)):
        try:
            owner = ndev.read_text().strip()
        except OSError:
            continue  # unpopulated GID table slots raise EINVAL on read
        if owner != interface:
            continue
        index = int(ndev.name)
        gid = (port / f"gids/{index}").read_text().strip()
        try:
            parsed = ipaddress.ip_address(gid.split("%", 1)[0])
        except ValueError:
            continue
        # Linux represents IPv4 RoCE GIDs as IPv4-mapped IPv6. Require v2,
        # since v1 and v2 entries can reference the same netdev.
        gid_type = (port / f"gid_attrs/types/{index}").read_text().strip()
        if isinstance(parsed, ipaddress.IPv6Address) and parsed.ipv4_mapped is not None and gid_type == "RoCE v2":
            matches.append((index, parsed.ipv4_mapped))
    if len(matches) != 1:
        raise ValueError(f"expected one IPv4 GID for {hca_path.name} on {interface}; found {matches}")
    return matches[0][0]


if __name__ == "__main__":
    import sys

    cfg = load_config(Path(__file__).resolve().parent)
    value = cfg
    for key in sys.argv[1].split("."):
        value = value[int(key)] if key.isdigit() else value[key]
    if isinstance(value, (dict, list)):
        raise SystemExit("config lookup must select a scalar value")
    print(value)
