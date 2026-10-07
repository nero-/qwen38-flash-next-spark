#!/usr/bin/env python3
"""Pinned single-Spark (TP1) runtime with composable, separately cached profiles.

launch.py start|stop|status|check [--profile tp1+mod...]
The profile defaults to selected-profile.txt, then node-config.json default_profile.
"""
import argparse
import json
import os
import socket
import subprocess
from pathlib import Path

from config import load_config, parse_profile

# Images verified to lack vLLM's NVFP4-CSF reader (nightly-20260923).
NO_CSF_IMAGES = {"sha256:5249a162cd39aa090e803243aef376e1f87f0fd4826f9249ea1ecd0c814b2c7e"}

p = argparse.ArgumentParser()
p.add_argument("action", choices=["start", "stop", "status", "check"])
p.add_argument("--profile", default=None)
a = p.parse_args()
home = Path.home()
config = load_config(Path(__file__).resolve().parent)
root = home / config["remote_root"]
model = str(root / "model-5500h")
images = config.get("experimental_images", {})
name = "qwen-tp1"
port = int(config["api_port"])
if a.action == "stop":
    subprocess.run(["docker", "stop", "--time", "60", name], check=True)
    raise SystemExit()
if a.action == "status":
    subprocess.run(["docker", "inspect", "--format", "{{.State.Status}} {{.State.StartedAt}}", name], check=True)
    raise SystemExit()

selection = root / "selected-profile.txt"
profile = a.profile or (selection.read_text().strip() if selection.exists() else config["default_profile"])
try:
    parsed = parse_profile(profile, images)
except ValueError as exc:
    p.error(str(exc))
flags = parsed["flags"]
image = images[parsed["image"]] if parsed["image"] else config["image"]
mounts = [model]
quantization, load_format = "modelopt_mixed", "b12x"
if "csf" in flags:
    # NVFP4-CSF: vLLM serves a metadata directory whose config points at the
    # compressed checkpoint (prepare_csf.py builds it). Needs a CSF-capable image.
    if image.split("@")[-1] in NO_CSF_IMAGES:
        p.error("csf needs an image with the NVFP4-CSF reader; this image has none (add an image modifier)")
    model = str(root / "serve-csf")
    mounts = [model, str(root / "model-csf")]
    quantization = load_format = "nvfp4_csf"
depth = parsed["mtp"] or 3
seqs = parsed["seqs"] or 16
if "resident" in flags and parsed["kv"] is None:
    p.error("resident PLE leaves ~27 GiB less memory; select an explicit +kvN (4 was validated)")
kv_gib = parsed["kv"] or int(config["kv_cache_gib"])
cache = root / f"cache-{profile.replace('+', '_')}"

env = dict(
    HOME=str(home), CUDA_HOME="/usr/local/cuda",
    TRITON_CACHE_DIR=f"{home}/.cache/triton",
    CUDA_CACHE_PATH=f"{home}/.cache/nv", XDG_CONFIG_HOME=f"{home}/.cache/config",
    TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas", CUDA_VISIBLE_DEVICES="0",
    CUTE_DSL_ARCH="sm_121a", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
    SAFETENSORS_FAST_GPU="1", OMP_NUM_THREADS="16",
    VLLM_WORKER_MULTIPROC_METHOD="spawn", HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1", VLLM_PLUGINS="b12x_loader",
    VLLM_SSM_CONV_STATE_LAYOUT="DS", VLLM_USE_AOT_COMPILE="1",
    VLLM_USE_MEGA_AOT_ARTIFACT="1", VLLM_USE_V2_MODEL_RUNNER="1",
    VLLM_MXFP8_LM_HEAD="1" if "mxhead" in flags else "0",
    VLLM_LM_HEAD_A16="1", VLLM_MTP_NVFP4_LM_HEAD="1",
    VLLM_QWEN3_8_FLASH_NEXT_OVERLAP="1", VLLM_QWEN3_8_FLASH_NEXT_MTP_COMPACT="1",
    VLLM_GDN_SPEC_DECODE_METADATA_FASTPATH="1", B12X_POLICY_MODE="auto",
)
if "resident" in flags:
    env["VLLM_PLE_CPU_OFFLOAD"] = "0"
else:
    # Read the single PLE layer's rows from the checkpoint with io_uring. This frees
    # ~27 GiB of the unified pool for KV; TP1 A/Bs found no decode-speed cost.
    env["VLLM_PLE_TABLE_MEMORY"] = "disk"

spec = dict(method="mtp", num_speculative_tokens=depth)
if "greedyspec" not in flags:
    spec.update(draft_sample_method="probabilistic", rejection_sample_method="block")
compilation = dict(pass_config=dict(fuse_act_quant=True))
if "cg4" in flags:
    # MTP verify batches are multiples of depth+1; capture each exactly.
    width = depth + 1
    compilation["cudagraph_capture_sizes"] = sorted({1, 2, *range(width, seqs * width + 1, width)})

server = ["-m", "vllm.entrypoints.cli.main", "serve", model,
    "--served-model-name", "qwen3.8-flash-next-4p89bpw", "--host", "0.0.0.0",
    "--port", str(port), "--trust-remote-code", "--tensor-parallel-size", "1",
    "--pipeline-parallel-size", "1", "--mamba-cache-mode", "align",
    "--enable-prefix-caching", "--enable-chunked-prefill", "--dtype", "bfloat16",
    "--kv-cache-dtype", "fp8", "--quantization", quantization, "--block-size", "16",
    "--load-format", load_format, "--kv-cache-memory-bytes", str(kv_gib * 2**30),
    "--max-model-len", "262144", "--max-num-seqs", str(seqs),
    "--max-num-batched-tokens", "8192", "--speculative-config", json.dumps(spec),
    "--mamba-ssm-cache-dtype", "float32" if "fp32ssm" in flags else "bfloat16",
    "--gdn-prefill-backend", "flashinfer", "--gdn-decode-kernel", "cuda",
    "--linear-backend", "b12x", "--moe-backend", "b12x",
    "--no-enable-flashinfer-autotune", "--mm-encoder-tp-mode", "data",
    "--mm-processor-cache-gb", "0", "--limit-mm-per-prompt", "{}",
    "--reasoning-parser", "qwen3", "--tool-call-parser", "qwen3_xml",
    "--enable-auto-tool-choice", "--compilation-config", json.dumps(compilation),
    "--max-cudagraph-capture-size", str(seqs * (depth + 1))]
command = ["docker", "run", "--detach", "--name", name, "--gpus", "all",
    "--network", "host", "--ipc", "host", "--security-opt", "seccomp=unconfined",
    "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
    "--user", f"{os.getuid()}:{os.getgid()}"]
for path in mounts:
    command += ["--mount", f"type=bind,src={path},dst={path},readonly"]
command += ["--mount", f"type=bind,src={cache},dst={home}/.cache", "--entrypoint", "/usr/bin/python3"]
for key, value in env.items():
    command += ["--env", f"{key}={value}"]
command += [image, *server]
if a.action == "check":
    print(json.dumps(command, indent=2))
    raise SystemExit()

assert (Path(model) / "config.json").exists(), f"Checkpoint missing: {model}"
cache.mkdir(parents=True, exist_ok=True)
(root / "evidence").mkdir(exist_ok=True)
old = subprocess.run(["docker", "inspect", name], capture_output=True, text=True)
if old.returncode == 0:
    info = json.loads(old.stdout)[0]
    expected = json.loads(subprocess.check_output(["docker", "image", "inspect", image]))[0]["Id"]
    actual_env = set(info["Config"]["Env"])
    assert info["Image"] == expected and info["Config"]["Cmd"] == server \
        and all(f"{k}={v}" in actual_env for k, v in env.items()), \
        "Existing qwen-tp1 container differs from the selected profile; use the profile switch"
    if not info["State"]["Running"]:
        subprocess.run(["docker", "start", name], check=True)
else:
    with socket.socket() as sock:
        assert sock.connect_ex(("127.0.0.1", port)) != 0, f"Port {port} is occupied"
    subprocess.run(command, check=True)
(root / "evidence" / f"{name}-command.json").write_text(json.dumps(command, indent=2))
