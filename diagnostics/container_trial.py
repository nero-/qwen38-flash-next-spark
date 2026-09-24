#!/usr/bin/env python3
"""Start the pinned eugr image with the source baseline's TP1 settings.

Run on Spark with the qwen38 venv after stopping the existing idle server.
The checkpoint is read-only; compiled caches are isolated from the host build.
"""
import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess

IMAGE = "eugr/spark-vllm-b12x@sha256:5249a162cd39aa090e803243aef376e1f87f0fd4826f9249ea1ecd0c814b2c7e"
parser = argparse.ArgumentParser()
parser.add_argument("--tag", default="eugr-20260923")
parser.add_argument("--check", action="store_true")
parser.add_argument("--gdn-decode-kernel", choices=["cuda", "b12x"], default="cuda",
    help="b12x selects paired b12x prefill/decode with required FP32 state")
args = parser.parse_args()
if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", args.tag):
    parser.error("tag must contain lowercase letters, digits or hyphens (max 80 characters)")
home = Path.home()
model = str(home / "models/Qwen3.8-Flash-Next-NVFP4")
root = home / "bench/tp1-opt-20260922"
cache = home / ".cache/qwen38-eugr-20260923"
name = "qwen38-" + args.tag
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
    VLLM_MXFP8_LM_HEAD="1", VLLM_LM_HEAD_A16="1", VLLM_MTP_NVFP4_LM_HEAD="1",
    VLLM_QWEN3_8_FLASH_NEXT_OVERLAP="1", VLLM_QWEN3_8_FLASH_NEXT_MTP_COMPACT="1",
    VLLM_GDN_SPEC_DECODE_METADATA_FASTPATH="1", B12X_POLICY_MODE="auto",
    VLLM_PLE_TABLE_MEMORY="disk",
)
server = ["-m", "vllm.entrypoints.cli.main", "serve", model,
    "--served-model-name", "qwen3.8-flash-next-4p89bpw", "--host", "0.0.0.0",
    "--port", "8000", "--trust-remote-code", "--tensor-parallel-size", "1",
    "--pipeline-parallel-size", "1", "--mamba-cache-mode", "align",
    "--enable-prefix-caching", "--enable-chunked-prefill", "--dtype", "bfloat16",
    "--kv-cache-dtype", "fp8", "--quantization", "modelopt_mixed", "--block-size", "16",
    "--load-format", "b12x", "--kv-cache-memory-bytes", "21474836480",
    "--max-model-len", "262144", "--max-num-seqs", "16",
    "--max-num-batched-tokens", "8192", "--speculative-config",
    '{"method":"mtp","num_speculative_tokens":3}',
    "--mamba-ssm-cache-dtype", "float32" if args.gdn_decode_kernel == "b12x" else "bfloat16",
    "--gdn-prefill-backend", "b12x" if args.gdn_decode_kernel == "b12x" else "flashinfer",
    "--gdn-decode-kernel", args.gdn_decode_kernel, "--linear-backend", "b12x", "--moe-backend", "b12x",
    "--no-enable-flashinfer-autotune", "--mm-encoder-tp-mode", "data",
    "--mm-processor-cache-gb", "0", "--limit-mm-per-prompt", "{}",
    "--reasoning-parser", "qwen3", "--tool-call-parser", "qwen3_xml",
    "--enable-auto-tool-choice", "--compilation-config", '{"pass_config":{"fuse_act_quant":true}}']
command = ["docker", "run", "--detach", "--name", name, "--gpus", "all",
    "--network", "host", "--ipc", "host", "--security-opt", "seccomp=unconfined",
    "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
    "--user", f"{os.getuid()}:{os.getgid()}",
    "--mount", f"type=bind,src={model},dst={model},readonly",
    "--mount", f"type=bind,src={cache},dst={home}/.cache",
    "--entrypoint", "/usr/bin/python3"]
for key, value in env.items():
    command += ["--env", f"{key}={value}"]
command += [IMAGE, *server]
if args.check:
    print(json.dumps(command, indent=2))
    raise SystemExit()
record_path = root / (args.tag + "-launch.json")
if record_path.exists():
    raise RuntimeError(f"Preserve prior trial: {record_path}")
with socket.socket() as check:
    if check.connect_ex(("127.0.0.1", 8000)) == 0:
        raise RuntimeError("Port 8000 already in use; stop the idle server first")
cache.mkdir(parents=True, exist_ok=True)
root.mkdir(parents=True, exist_ok=True)
subprocess.run(["docker", "image", "inspect", IMAGE], check=True, stdout=subprocess.DEVNULL)
container_id = subprocess.check_output(command, text=True).strip()
info = json.loads(subprocess.check_output(["docker", "inspect", container_id]))[0]
record = dict(mode="container", container_id=container_id, container_name=name,
    started_at=info["State"]["StartedAt"],
    image=IMAGE, image_id=info["Image"], launcher_pid=info["State"]["Pid"],
    command=command, load_format="b12x", env=env)
record_path.write_text(json.dumps(record, indent=2))
with (root / (args.tag + "-server.log")).open("x") as log:
    subprocess.Popen(["docker", "logs", "--follow", container_id], stdout=log,
        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
print(json.dumps(record), flush=True)
