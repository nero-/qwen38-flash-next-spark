#!/usr/bin/env python3
"""Pinned two-Spark Qwen runtime. Run on each rank; no source overrides."""
import argparse, json, os, subprocess, socket
from pathlib import Path
IMAGE = "eugr/spark-vllm-b12x@sha256:5249a162cd39aa090e803243aef376e1f87f0fd4826f9249ea1ecd0c814b2c7e"
p = argparse.ArgumentParser()
p.add_argument("action", choices=["start", "stop", "status", "check", "rdma"])
p.add_argument("--rank", type=int, choices=[0,1], required=True)
a = p.parse_args()
home = Path.home()
root = home / "builds/qwen-tp2"
model = str(root / "model")
cache = root / "cache"
name = "qwen-tp2"
rank = a.rank
ip = ["10.100.168.2", "10.100.168.1"][rank]
nic = ["enp1s0f1np1", "enp1s0f0np0"][rank]
hca = ["rocep1s0f1,roceP2p1s0f1", "rocep1s0f0,roceP2p1s0f0"][rank]
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
    VLLM_MXFP8_LM_HEAD="0", VLLM_LM_HEAD_A16="1", VLLM_MTP_NVFP4_LM_HEAD="1",
    VLLM_QWEN3_8_FLASH_NEXT_OVERLAP="1", VLLM_QWEN3_8_FLASH_NEXT_MTP_COMPACT="1",
    VLLM_GDN_SPEC_DECODE_METADATA_FASTPATH="1", B12X_POLICY_MODE="auto",
    VLLM_PLE_CPU_OFFLOAD="0",
)
server = ["-m", "vllm.entrypoints.cli.main", "serve", model,
    "--served-model-name", "qwen3.8-flash-next-4p89bpw", "--host", "0.0.0.0",
    "--port", "8000", "--trust-remote-code", "--tensor-parallel-size", "2",
    "--pipeline-parallel-size", "1", "--mamba-cache-mode", "align",
    "--enable-prefix-caching", "--enable-chunked-prefill", "--dtype", "bfloat16",
    "--kv-cache-dtype", "fp8", "--quantization", "modelopt_mixed", "--block-size", "16",
    "--load-format", "b12x", "--kv-cache-memory-bytes", "21474836480",
    "--max-model-len", "262144", "--max-num-seqs", "16",
    "--max-num-batched-tokens", "8192", "--speculative-config",
    '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic","rejection_sample_method":"block"}',
    "--mamba-ssm-cache-dtype", "bfloat16",
    "--gdn-prefill-backend", "flashinfer",
    "--gdn-decode-kernel", "cuda", "--linear-backend", "b12x", "--moe-backend", "b12x",
    "--no-enable-flashinfer-autotune", "--mm-encoder-tp-mode", "data",
    "--mm-processor-cache-gb", "0", "--limit-mm-per-prompt", "{}",
    "--reasoning-parser", "qwen3", "--tool-call-parser", "qwen3_xml",
    "--enable-auto-tool-choice", "--compilation-config", '{"pass_config":{"fuse_act_quant":true}}']

env.update(VLLM_HOST_IP=ip, NCCL_SOCKET_IFNAME=nic, GLOO_SOCKET_IFNAME=nic,
    NCCL_IB_HCA=hca, B12X_ROCE_HCA=hca, NCCL_IB_GID_INDEX="3",
    NCCL_IB_TC="106", B12X_ROCE_TRAFFIC_CLASS="106", NCCL_IB_MERGE_NICS="1",
    NCCL_NET_PLUGIN="none", NCCL_DEBUG="INFO", VLLM_ENABLE_PCIE_ALLREDUCE="0",
    VLLM_ENABLE_ROCE_ALLREDUCE="1", VLLM_ROCE_ALLREDUCE_MAX_SIZE="2MB",
    VLLM_ROCE_ALLGATHER_MAX_SIZE="16MB", B12X_ROCE_CACHE_DIR=f"{home}/.cache/b12x-roce")
server += ["--distributed-executor-backend", "mp", "--data-parallel-backend", "mp",
    "--gpu-memory-utilization", "0.80",
    "--nnodes", "2", "--node-rank", str(rank), "--master-addr", "10.100.168.2",
    "--master-port", "29638", "--max-cudagraph-capture-size", "64"]
if rank:
    server += ["--headless"]
if a.action == "rdma":
    name = "qwen-tp2-rdma-check"
    env.update(RANK=str(rank), WORLD_SIZE="2", MASTER_ADDR="10.100.168.2", MASTER_PORT="29639")
    server = [str(root / "rdma_check.py")]
command = ["docker", "run", "--detach", "--name", name, "--gpus", "all",
    "--network", "host", "--ipc", "host", "--security-opt", "seccomp=unconfined",
    "--device", "/dev/infiniband", "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
    "--user", f"{os.getuid()}:{os.getgid()}",
    "--mount", f"type=bind,src={model},dst={model},readonly",
    "--mount", f"type=bind,src={cache},dst={home}/.cache",
    "--mount", f"type=bind,src={root}/rdma_check.py,dst={root}/rdma_check.py,readonly",
    "--entrypoint", "/usr/bin/python3"]
for k,v in env.items(): command += ["--env", f"{k}={v}"]
command += [IMAGE, *server]
if a.action == "check":
    print(json.dumps(command, indent=2)); raise SystemExit()
if a.action == "stop":
    subprocess.run(["docker", "stop", "--time", "60", name], check=True); raise SystemExit()
if a.action == "status":
    subprocess.run(["docker", "inspect", "--format", "{{.State.Status}} {{.State.StartedAt}}", name], check=True); raise SystemExit()
cache.mkdir(parents=True, exist_ok=True)
(root / "evidence").mkdir(exist_ok=True)
assert (Path(model) / "config.json").exists(), "Checkpoint missing"
old = subprocess.run(["docker", "inspect", name], capture_output=True, text=True)
if old.returncode == 0:
    info=json.loads(old.stdout)[0]
    expected=json.loads(subprocess.check_output(["docker","image","inspect",IMAGE]))[0]["Id"]
    actual_env=set(info["Config"]["Env"])
    assert info["Image"] == expected and info["Config"]["Cmd"] == server and all(f"{k}={v}" in actual_env for k,v in env.items()), "Existing container differs; inspect before recreating"
    if not info["State"]["Running"]: subprocess.run(["docker","start",name],check=True)
else:
    if rank == 0 and a.action == "start":
        with socket.socket() as sock:
            assert sock.connect_ex(("127.0.0.1",8000)) != 0, "Port 8000 is occupied"
    subprocess.run(command, check=True)
(root / "evidence" / f"{name}-rank{rank}-command.json").write_text(json.dumps(command,indent=2))
