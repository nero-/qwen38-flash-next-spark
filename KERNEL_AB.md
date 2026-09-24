# Kernel-only comparison

The current Spark runs `7.0.0-1019-nvidia`, NVIDIA driver `580.178.04`, Secure Boot enabled. Its existing boot parameters already include `kho=off`. NVIDIA's September advisory confirms problems with this kernel for multi-node NCCL/RoCE workloads; it does not establish a TP1 inference slowdown on this machine.

Source: [NVIDIA advisory and reports](https://forums.developer.nvidia.com/t/dgx-spark-regression-kernel-7-0-0-1019-nvidia-causes-nccl-roce-ibv-reg-mr-iova2-enomem-6-17-0-1032-works/383023).

Only kernels 7.0 and 6.11 are installed initially. `6.17.0-1032-nvidia` is available from the configured Ubuntu archive, including signed NVIDIA modules built for the existing 580.178.04 driver. The reviewed apt simulation adds five packages, upgrades none, and removes none.

After all baseline tests finish, run from the Mac:

```bash
ssh -t spark-r0 'sudo bash ~/projects/qwen38-flash-next-spark/diagnostics/kernel_617_once.sh'
```

The script backs up GRUB configuration/environment and package inventory under `/var/lib/qwen38-kernel-ab/`, validates the package plan, installs the pinned 6.17 packages, checks module version and signer, selects the actual generated GRUB entry for one boot, stops the idle model server, and reboots. It refuses to stop an API with active requests. It does not remove kernel 7.0, alter Secure Boot, replace driver userspace or change the permanent boot default. A subsequent normal reboot returns to the default 7.0 kernel unless that default is deliberately changed later.

After SSH returns, verify `uname -r` reports `6.17.0-1032-nvidia` and `nvidia-smi` still reports driver 580.178.04. Start a fresh trial of the same pinned container with the original CUDA-GDN/BF16 settings, and repeat the unmodified LIL matrix and Coding Peak. Keep MTP 3, full vocabulary, disk PLE, 262144 context, 20 GiB KV, max 16 sequences, and 8192-token chunks unchanged. Compare with `eugr-20260923-r2` on kernel 7.0; do not compare against the separate b12x/FP32 experiment when attributing a kernel effect.

The user ran the script successfully. The Spark booted into 6.17.0-1032-nvidia at 17:27:59 UTC on September 23. Driver 580.178.04 remained active, and the same-image comparison completed. The permanent boot default remains unchanged.

The subsequent reboot returned r0 to 7.0.0-1019-nvidia. Both TP2 ranks use
7.0 with driver 580.178.04. The paired RDMA correctness check passed on both
active links; a kernel rollback was not needed for this bring-up. Completed
TP1 performance results are recorded in [BENCHMARK.md](BENCHMARK.md).
