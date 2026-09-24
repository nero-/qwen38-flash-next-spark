#!/usr/bin/env python3
"""Pinned two-Spark runtime with explicit, separately cached experimental profiles."""
import argparse, json, os, subprocess, socket, sys
from pathlib import Path
from config import load_config, selected_rails, find_ipv4_gid
p = argparse.ArgumentParser()
p.add_argument("action", choices=["start", "stop", "status", "check", "rdma"])
p.add_argument("--rank", type=int, choices=[0,1], required=True)
p.add_argument("--profile", default=None)
p.add_argument("--cables", type=int, choices=[1,2], default=None,
               help="preflight this cable mode without changing selected-cables.txt")
a = p.parse_args()
home = Path.home()
script_root = Path(__file__).resolve().parent
config = load_config(script_root)
root = home / config["remote_root"]
model = str(root / "model")
IMAGE = config["image"]
profiles = {
    "baseline": (False, False, 3, False),
    "k20": (False, True, 3, False),
    "hc": (True, False, 3, False),
    "hc-adaptive": (True, False, 3, False),
    "hc-prefill": (True, False, 3, False),
    "hc-prefill-dynamic": (True, False, 3, False),
    "hc-mtp4": (True, False, 4, False),
    "hc-mtp5": (True, False, 5, False),
    "hc-k20": (True, True, 3, False),
    "hc-k20-mtp4": (True, True, 4, False),
    "hc-k20-mtp5": (True, True, 5, False),
    "hc-k20-mxfp8-w13": (True, True, 3, True),
    "hc-k20-mtp5-mxfp8-w13": (True, True, 5, True),
    "k20-mtp4": (False, True, 4, False),
    "k20-mtp5": (False, True, 5, False),
    "k20-mxfp8-w13": (False, True, 3, True),
}
selection = root / "selected-profile.txt"
name = "qwen-tp2"
rank = a.rank
if a.action == "stop":
    subprocess.run(["docker", "stop", "--time", "60", name], check=True)
    raise SystemExit()
if a.action == "status":
    subprocess.run(["docker", "inspect", "--format", "{{.State.Status}} {{.State.StartedAt}}", name], check=True)
    raise SystemExit()
profile = a.profile or (selection.read_text().strip() if selection.exists() else config["default_profile"])
trace = profile.endswith("-trace")
base_profile = profile.removesuffix("-trace") if trace else profile
if base_profile not in profiles: p.error(f"Unknown profile: {profile}")
hybrid, topk, depth, quant_experiment = profiles[base_profile]
cache = root / ("cache" if base_profile == "baseline" else f"cache-{base_profile}")
rank_config = config["ranks"][rank]
ip = rank_config["ip"]
nic = rank_config["socket_interface"]
cable_selection = root / "selected-cables.txt"
cables = a.cables if a.cables is not None else (int(cable_selection.read_text().strip()) if cable_selection.exists() else config["default_cables"])
if cables not in (1, 2): p.error("selected-cables.txt must contain 1 or 2")
rails = selected_rails(config, rank, cables)
hca = ",".join(rail["hca"] for rail in rails)
gid_indexes = [find_ipv4_gid(Path("/sys/class/infiniband") / rail["hca"], rail["interface"]) for rail in rails]
if len(set(gid_indexes)) != 1:
    p.error(f"Selected HCAs resolve to different GID indexes {gid_indexes}; backend requires one shared NCCL_IB_GID_INDEX")
gid_index = str(gid_indexes[0])
for rail in rails:
    selected_nic, selected_hca = rail["interface"], rail["hca"]
    net = Path("/sys/class/net") / selected_nic
    if not net.exists() or (net / "operstate").read_text().strip() != "up":
        p.error(f"DAC link down or missing: {selected_nic}")
    port = Path("/sys/class/infiniband") / selected_hca / "ports/1"
    if (port / f"gid_attrs/ndevs/{gid_index}").read_text().strip() != selected_nic:
        p.error(f"GID {gid_index} on {selected_hca} does not belong to {selected_nic}")
    try:
        gid_address = (port / f"gids/{gid_index}").read_text().strip()
    except OSError as exc:
        p.error(f"Missing RoCE GID {gid_index} for {selected_hca}: {exc}")
    if not int(gid_address.replace(":", ""), 16):
        p.error(f"Missing RoCE GID: {selected_hca}")
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
    VLLM_GDN_SPEC_DECODE_METADATA_FASTPATH="1",
    VLLM_PLE_CPU_OFFLOAD="0",
)
server = ["-m", "vllm.entrypoints.cli.main", "serve", model,
    "--served-model-name", "qwen3.8-flash-next-4p89bpw", "--host", "0.0.0.0",
    "--port", str(config["api_port"]), "--trust-remote-code", "--tensor-parallel-size", "2",
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
    NCCL_IB_HCA=hca, B12X_ROCE_HCA=hca, NCCL_IB_GID_INDEX=gid_index,
    NCCL_IB_TC="106", B12X_ROCE_TRAFFIC_CLASS="106", NCCL_IB_MERGE_NICS="1",
    NCCL_NET_PLUGIN="none", NCCL_DEBUG="INFO", VLLM_ENABLE_PCIE_ALLREDUCE="0",
    VLLM_ENABLE_ROCE_ALLREDUCE="1", VLLM_ROCE_ALLREDUCE_MAX_SIZE="2MB",
    VLLM_ROCE_ALLGATHER_MAX_SIZE="16MB", B12X_ROCE_CACHE_DIR=f"{home}/.cache/b12x-roce")
server += ["--distributed-executor-backend", "mp", "--data-parallel-backend", "mp",
    "--gpu-memory-utilization", "0.80",
    "--nnodes", "2", "--node-rank", str(rank), "--master-addr", config["ranks"][0]["ip"],
    "--master-port", str(config["master_port"]), "--max-cudagraph-capture-size", str(16 * (depth + 1))]
server[server.index("--speculative-config")+1] = json.dumps(dict(method="mtp", num_speculative_tokens=depth,
    draft_sample_method="probabilistic", rejection_sample_method="block"))
if hybrid:
    env.update(VLLM_QWEN3_8_FLASH_NEXT_HC_TP="0", VLLM_QWEN3_8_HC_PREFILL_MODE="shard")
if base_profile == "hc-adaptive": env["QWEN_HC_CHANNEL_DECODE_MAX_TOKENS"] = "4"
if base_profile.startswith("hc-prefill"): env["QWEN_HC_CHANNEL_DECODE"] = "0"
if base_profile == "hc-prefill-dynamic": env["B12X_MICRO_DYNAMIC_CUTOVER_PAIRS"] = "0"
if topk: env["QWEN_MTP_DRAFT_TOP_K"] = "20"
if quant_experiment:
    env.update(VLLM_MXFP8_LM_HEAD="1", VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE="w13")
if trace:
    server += ["--profiler-config", json.dumps(dict(profiler="torch",
        torch_profiler_dir=f"{home}/.cache/trace", torch_profiler_with_stack=False,
        torch_profiler_record_shapes=False, ignore_frontend=True,
        max_iterations=8))]
if rank:
    server += ["--headless"]
if a.action == "rdma":
    name = "qwen-tp2-rdma-check"
    env.update(RANK=str(rank), WORLD_SIZE="2", MASTER_ADDR=config["ranks"][0]["ip"], MASTER_PORT=str(config["master_port"] + 1))
    server = [str(root / "rdma_check.py")]
command = ["docker", "run", "--detach", "--name", name, "--gpus", "all",
    "--network", "host", "--ipc", "host", "--security-opt", "seccomp=unconfined",
    "--device", "/dev/infiniband", "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
    "--user", f"{os.getuid()}:{os.getgid()}",
    "--mount", f"type=bind,src={model},dst={model},readonly",
    "--mount", f"type=bind,src={cache},dst={home}/.cache",
    "--mount", f"type=bind,src={root}/rdma_check.py,dst={root}/rdma_check.py,readonly",
    "--entrypoint", "/usr/bin/python3"]
candidate = root / "optimization/candidate"
package = "/usr/local/lib/python3.12/dist-packages/vllm"
overrides = {}
if hybrid:
    for filename in ("hyperconnection.py", "model.py", "hc_prefill.py"):
        overrides[filename] = f"{package}/models/qwen4_exp/nvidia/{filename}"
    overrides["qwen3_next.py"] = f"{package}/model_executor/models/qwen3_next.py"
if topk: overrides["speculator.py"] = f"{package}/v1/worker/gpu/spec_decode/speculator.py"
for filename, target in overrides.items():
    command += ["--mount", f"type=bind,src={candidate / filename},dst={target},readonly"]
if overrides: env["PYTHONDONTWRITEBYTECODE"] = "1"
for k,v in env.items(): command += ["--env", f"{k}={v}"]
command += [IMAGE, *server]
# Verify the intended source bytes before creating any experimental container.
if overrides:
    import hashlib
    manifest = json.loads((root / "optimization/candidate-sha256.json").read_text())
    for filename in overrides:
        assert hashlib.sha256((candidate / filename).read_bytes()).hexdigest() == manifest[f"candidate/{filename}"], f"Source drift: {filename}"
if a.action == "check":
    print(json.dumps(command, indent=2)); raise SystemExit()
# Apply runtime MTU only to interfaces selected by the cable mode. This is
# intentionally ephemeral; no persistent host network files are changed.
if a.action == "start":
    for rail in rails:
        selected_nic = rail["interface"]
        net = Path("/sys/class/net") / selected_nic
        if (net / "mtu").read_text().strip() != "9000":
            subprocess.run([
                "docker", "run", "--rm", "--network", "host", "--cap-add", "NET_ADMIN", "--user", "0",
                "--mount", "type=bind,src=/usr/sbin/ip,dst=/host-ip,readonly",
                "--mount", "type=bind,src=/usr/lib/aarch64-linux-gnu,dst=/host-lib,readonly",
                "--entrypoint", "/host-lib/ld-linux-aarch64.so.1", IMAGE,
                "--library-path", "/host-lib", "/host-ip", "link", "set", "dev", selected_nic, "mtu", "9000",
            ], check=True)
        assert (net / "mtu").read_text().strip() == "9000", f"MTU not applied: {selected_nic}"

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
