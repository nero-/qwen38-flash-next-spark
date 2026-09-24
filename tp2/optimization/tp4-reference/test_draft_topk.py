"""Truncated proposals must retain target mass outside their support.

Use the installed GPU sampler and upstream narrow-vocabulary distribution
oracle with actual draft RNG. The proposal filter is imported from the candidate.
"""
import importlib.util
import sys

import pytest
import torch
import test_rejection_sampler_utils as oracle
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

spec = importlib.util.spec_from_file_location("candidate_speculator", "/tmp/speculator.py")
candidate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = candidate
spec.loader.exec_module(candidate)


@pytest.mark.parametrize("steps", [1, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("temperature", [0.6, 1.0])
@pytest.mark.parametrize("top_k", [3, 8])
@pytest.mark.parametrize("block", [False, True])
def test_target_distribution_with_truncated_proposal(steps, dtype, temperature, top_k, block):
    torch.manual_seed(42)
    target = torch.randn(16, device="cuda", dtype=torch.float32)
    draft = candidate._limit_draft_logits((-target).to(dtype)[None], top_k)[0]
    assert torch.isneginf(draft).sum() == 16 - top_k
    target = target / temperature
    inputs = oracle._build_rejection_sample_inputs(
        target, draft, steps, temperature=temperature,
        num_trials=oracle.NARROW_NUM_TRIALS,
    )
    inputs["draft_sampled"] = oracle._gumbel_drafted_tokens(
        inputs, draft, oracle.NARROW_NUM_TRIALS, steps,
    )
    sampled, counts = oracle.rejection_sample(
        **inputs, num_speculative_steps=steps, use_block_verification=block,
    )
    assert (counts >= 1).all()
    for position in range(steps + 1):
        oracle._assert_distribution_match(
            sampled[counts > position, position], target.softmax(0), "cuda",
            label=f"position {position}",
        )


@pytest.mark.parametrize("temperature", [0.0, 0.6, 1.0])
def test_cached_proposal_and_graph_replay(temperature):
    # Most vocabulary blocks contain only -inf after top-k truncation.
    torch.manual_seed(941)
    raw = torch.randn(4, 151936, device="cuda", dtype=torch.float32)
    mapping = torch.tensor([2, 0, 3, 1], device="cuda", dtype=torch.int32)
    temps = torch.full((4,), temperature, device="cuda")
    seeds = torch.arange(4, device="cuda", dtype=torch.int64)
    pos = torch.arange(4, device="cuda", dtype=torch.int32)
    col = torch.tensor(1, device="cuda", dtype=torch.int32)
    cache = torch.empty(4, 3, raw.shape[1], device="cuda", dtype=raw.dtype)
    def sample():
        proposal = candidate._limit_draft_logits(raw, 20)
        return gumbel_sample(
            proposal, mapping, temps, seeds, pos, is_drafting=True,
            apply_temperature=True, logits_cache=cache, logits_cache_col=col,
        )
    expected = raw.argsort(dim=-1, descending=True)[:, :20]
    sample()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = sample()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    cached = cache[mapping.long(), 1]
    assert torch.isfinite(cached).sum(-1).eq(20).all()
    assert (result[:, None] == expected).any(-1).all()
    torch.testing.assert_close(cached.gather(1, expected), raw.gather(1, expected))
    if temperature == 0:
        assert result.eq(raw.argmax(-1)).all()
