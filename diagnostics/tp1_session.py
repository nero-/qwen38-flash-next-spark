#!/usr/bin/env python3
"""Launch reversible TP1 profiling/capacity experiments on spark-r0.

Run with the existing qwen38 virtualenv. Never alters model or engine sources.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

HOME = Path.home()
MODEL = str(HOME / "models/Qwen3.8-Flash-Next-NVFP4")
ROOT = HOME / "bench/tp1-opt-20260922"


def servers():
    found = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            argv = (proc / "cmdline").read_bytes().split(b"\0")
            if b"vllm.entrypoints.cli.main" in argv and MODEL.encode() in argv:
                found.append(int(proc.name))
        except (OSError, ValueError):
            pass
    return found


def stop():
    with urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=5) as res:
        for line in res.read().decode().splitlines():
            if line.startswith(("vllm:num_requests_running{", "vllm:num_requests_waiting{")):
                if float(line.rsplit(" ", 1)[1]) != 0:
                    raise RuntimeError("Active inference requests; refusing to stop")
    pids = servers()
    if len(pids) != 1:
        raise RuntimeError(f"Expected one exact model server, found {pids}")
    pid = pids[0]
    stamp = str(time.time_ns())
    (ROOT / (stamp + "-previous-command.bin")).write_bytes(Path(f"/proc/{pid}/cmdline").read_bytes())
    os.kill(pid, signal.SIGTERM)
    for _ in range(60):
        if not servers():
            print(json.dumps({"stopped": pid}), flush=True)
            return
        time.sleep(1)
    raise RuntimeError("Server did not exit gracefully; no force-kill attempted")


def start(mode, tag=None, omp_threads=None, batched_tokens=None,
          load_format="b12x", cuda_home="/usr/local/cuda-13.3"):
    if servers():
        raise RuntimeError("Existing server must be stopped first")
    tag = tag or mode
    env = os.environ.copy()
    env.update(PATH=f"{HOME}/venvs/qwen38/bin:{cuda_home}/bin:/usr/local/bin:/usr/bin:/bin",
               CUDA_HOME=cuda_home, TRITON_PTXAS_PATH=f"{cuda_home}/bin/ptxas",
               PYTHON_BIN=str(HOME / "venvs/qwen38/bin/python"), MODEL_PATH=MODEL,
               VLLM_PLE_TABLE_MEMORY="disk", B12X_POLICY_MODE="auto",
               MAX_NUM_SEQS="16", KV_CACHE_MEMORY_BYTES=str(20 * 2**30),
               MAX_MODEL_LEN="262144", MAX_NUM_BATCHED_TOKENS="8192",
               CUDAGRAPH_CAPTURE_SIZES="auto", NUM_SPECULATIVE_TOKENS="3",
               OMP_NUM_THREADS="16",
               LOG_FILE=str(ROOT / (tag + "-server.log")))
    if mode in ("resident", "lean-disk"):
        env.update(MAX_NUM_SEQS="1",
                   KV_CACHE_MEMORY_BYTES=str(4 * 2**30), MAX_MODEL_LEN="262144",
                   MAX_NUM_BATCHED_TOKENS="2048")
    if mode == "resident":
        # The environment enum accepts only ram/disk. None plus offload=0
        # selects device storage in resolve_ple_table_memory.
        env.pop("VLLM_PLE_TABLE_MEMORY", None)
        env["VLLM_PLE_CPU_OFFLOAD"] = "0"
    if omp_threads is not None:
        env["OMP_NUM_THREADS"] = str(omp_threads)
    if batched_tokens is not None:
        env["MAX_NUM_BATCHED_TOKENS"] = str(batched_tokens)
    command = ["bash", "serve-qwen38-flash-next-nvfp4-tp1.sh", "--",
               "--mamba-ssm-cache-dtype", "bfloat16", "--gdn-prefill-backend", "flashinfer",
               "--gdn-decode-kernel", "cuda", "--load-format", load_format,
               "--limit-mm-per-prompt", "{}"]
    if mode == "profile":
        command += ["--profiler-config", json.dumps({"profiler": "cuda", "max_iterations": 8})]
        command = ["nsys", "profile", "--trace=cuda,nvtx,osrt", "--sample=none", "--cpuctxsw=none",
                   "--cuda-graph-trace=node", "--capture-range=cudaProfilerApi",
                   "--capture-range-end=repeat:2", "--kill=none",
                   "--output=" + str(ROOT / "disk-decode"), *command]
    subprocess.run([env["PYTHON_BIN"], "-c",
                    "import vllm.envs as e; print('PLE preflight:', e.VLLM_PLE_TABLE_MEMORY, e.VLLM_PLE_CPU_OFFLOAD)"],
                   cwd=HOME / "projects/vllm", env=env, check=True)
    log_path = ROOT / (tag + "-launch.log")
    if log_path.exists():
        raise RuntimeError(f"Preserve previous log before reusing mode: {log_path}")
    with log_path.open("w") as log:
        child = subprocess.Popen(command, cwd=HOME / "projects/vllm", env=env,
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    record = {"mode": mode, "launcher_pid": child.pid, "command": command,
              "load_format": load_format,
              "source_heads": {repo: subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], cwd=HOME / "projects" / repo,
                  text=True).strip() for repo in ("vllm", "b12x")},
              "env": {k: env.get(k) for k in ("VLLM_PLE_TABLE_MEMORY", "VLLM_PLE_CPU_OFFLOAD", "MAX_NUM_SEQS",
                      "KV_CACHE_MEMORY_BYTES", "MAX_MODEL_LEN", "MAX_NUM_BATCHED_TOKENS",
                      "NUM_SPECULATIVE_TOKENS", "OMP_NUM_THREADS", "CUDA_HOME")}}
    (ROOT / (tag + "-launch.json")).write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["stop", "profile", "resident", "lean-disk", "restore"])
    parser.add_argument("--tag")
    parser.add_argument("--omp-threads", type=int)
    parser.add_argument("--batched-tokens", type=int)
    parser.add_argument("--load-format", choices=["b12x", "instanttensor"], default="b12x")
    parser.add_argument("--cuda-home", default="/usr/local/cuda-13.3")
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.action == "stop":
        stop()
    else:
        start(args.action, args.tag, args.omp_threads, args.batched_tokens,
              args.load_format, args.cuda_home)
