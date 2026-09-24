# Deployment artifact provenance

- Model: [`local-inference-lab/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4) at immutable revision `7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd`. The repository's own model card, license file, and usage terms at that revision govern access and use. No model weights or download cache are stored in Git.
- Runtime: [`eugr/spark-vllm-b12x`](https://hub.docker.com/r/eugr/spark-vllm-b12x), selected by OCI digest in `cluster-config.json`. The image includes vLLM, CUDA/NVIDIA runtime components, and b12x; its upstream image/license notices govern its contents.
- Source overrides: files in `optimization/candidate/` are vLLM-derived and retain their Apache-2.0 SPDX notices. Their SHA-256 manifest is checked before an override profile can start. `baseline-source/` and `tp4-reference/` are research/reference files, not mounted by the selected default profile.
- Benchmark evidence: the JSON and text files in `results/tp2/` are receipts from the named deployment and tool versions. Machine-specific inspection outputs, full server logs, memory streams, and profiler traces are excluded from the release package; their local copies are not deleted.

The model card and OCI image documentation should be reviewed again for a new snapshot or image. A digest pins bytes; it does not grant rights to use or redistribute the artifact.
