"""Exercise the same sparse-proposal distribution oracle at higher MTP depths."""
import pytest
import torch
import test_draft_topk as base


@pytest.mark.parametrize("steps", [4, 5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("temperature", [0.6, 1.0])
@pytest.mark.parametrize("top_k", [3, 8])
@pytest.mark.parametrize("block", [False, True])
def test_higher_depth_distribution(steps, dtype, temperature, top_k, block):
    base.test_target_distribution_with_truncated_proposal(
        steps, dtype, temperature, top_k, block)
