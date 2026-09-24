# Qwen3.8-Flash-Next on two DGX Sparks

This repository packages a two-node vLLM deployment for DGX Spark. It preserves the full model vocabulary and checkpoint, BF16 target head and recurrent state, resident PLE, 20 GiB FP8 KV cache per rank, and 262,144-token context. The selected profile is `hc-adaptive` with MTP3; `hc` remains the original-HC backup. The example cluster selects two DAC cables.

The exact deployment image is pinned to `eugr/spark-vllm-b12x@sha256:5249a162cd39aa090e803243aef376e1f87f0fd4826f9249ea1ecd0c814b2c7e`. The model snapshot is pinned to `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` revision `7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd`. The current cluster passed its documented functional and performance checks; a clean installation on different hardware, universal quality equivalence, and production qualification have not been established.

Start with the [TP2 installation and operations guide](tp2/README.md). Its example `cluster-config.json` is where another cluster's SSH targets, direct-link addresses, NIC/HCA names, image, model snapshot and one/two-cable selection belong. The bootstrap copies the runtime package and verifies independently downloaded model files, but never starts serving.

Useful records: [cable comparison](tp2/optimization/CABLES.md), [HC dispatch results](tp2/optimization/HC-SEPARATION.md), and [retained evidence guide](results/README.md). TP1 material is preserved in [TP1_README_ARCHIVE.md](TP1_README_ARCHIVE.md) and [TP1_OPERATIONS_ARCHIVE.md](TP1_OPERATIONS_ARCHIVE.md) as historical context.
