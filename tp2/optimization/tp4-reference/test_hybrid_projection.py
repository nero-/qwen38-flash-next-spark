"""Check every TP partition against complete FP32 projection equations.

Collective inputs are checked independently for each rank. Owned rows must
never enter a channel collective, even when their count fits the decode buffer.
Existing HC kernel arithmetic is replaced by explicit PyTorch equations here;
this test isolates the new projection ownership and dispatch contract.
"""
import importlib.util
import math
import sys
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import vllm.model_executor.layers.linear as linear
import vllm.distributed.parallel_state as parallel_state

NAME = 'vllm.models.qwen4_exp.nvidia.hyperconnection_candidate'
spec = importlib.util.spec_from_file_location(NAME, '/tmp/hyperconnection.py')
hc = importlib.util.module_from_spec(spec)
sys.modules[NAME] = hc
spec.loader.exec_module(hc)

class Equations:
    @staticmethod
    def run_combine_norm(hidden, block, injection, weight, *, eps, plan):
        combined = (hidden.float().view(hidden.shape[0], 4, -1)
                    + block.float()[:, None, :]
                    * (2 * (injection.float() / 4).sigmoid())[:, :, None]
                    ).flatten(1).bfloat16()
        return combined, Equations.run_grouped_rmsnorm(
            combined, weight, eps=eps, binding=None)

    @staticmethod
    def run_grouped_rmsnorm(x, w, *, eps, binding):
        grouped = x.float().view(x.shape[0], 4, -1)
        y = grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + eps)
        return (y.flatten(1) * (1 + w.float())).to(x.dtype)

    @staticmethod
    def run_scaled_silu(x, *, binding):
        return F.silu(x.float() / 4).to(x.dtype)

    @staticmethod
    def run_gate_mean(x, gates, *, binding):
        y = x.float() * gates.float().sigmoid()
        return y.view(x.shape[0], 4, -1).mean(1).to(x.dtype)

@pytest.mark.parametrize('rows', [1, 4, 5, 6, 64, 96, 128, 256])
@pytest.mark.parametrize('use_combine', [False, True])
@pytest.mark.parametrize('pending', [False, True])
def test_channel_partitions_and_owned_rows(monkeypatch, rows, use_combine, pending):
    monkeypatch.setenv('VLLM_QWEN3_8_FLASH_NEXT_HC_TP', '0')
    monkeypatch.setenv('VLLM_QWEN3_8_HC_PREFILL_MODE', 'shard')
    monkeypatch.setattr(hc, 'get_tensor_model_parallel_world_size', lambda: 4)
    monkeypatch.setattr(linear, 'get_tensor_model_parallel_world_size', lambda: 4)
    monkeypatch.setattr(hc, '_hyperconnection_api', lambda: Equations)
    config = hc.HyperConnectionConfig(hc_count=4, hidden_size=2560,
        params_dtype=torch.bfloat16, hc_lowrank=320, rms_norm_eps=1e-6,
        hc_per_branch_norm=True)
    torch.manual_seed(317)
    x = torch.randn(rows, 10240, device='cuda', dtype=torch.bfloat16)
    down = torch.randn(336 if use_combine else 320, 10240,
        device='cuda', dtype=torch.bfloat16) / math.sqrt(10240)
    up = torch.randn(10240, 320, device='cuda', dtype=torch.bfloat16) / math.sqrt(320)
    norm = torch.randn(10240, device='cuda', dtype=torch.bfloat16) * .01
    block = torch.randn(rows, 2560, device='cuda', dtype=torch.bfloat16)
    previous = torch.randn(rows, 4, device='cuda', dtype=torch.bfloat16)
    if pending:
        expected_hidden = (x.float().view(rows, 4, 2560)
                           + 2 * block.float().unsqueeze(1)
                           / (1 + (-previous.float() / 4).exp()).unsqueeze(2))
        expected_hidden = expected_hidden.flatten(1).bfloat16()
    else:
        expected_hidden = x
    normalized = Equations.run_grouped_rmsnorm(
        expected_hidden, norm, eps=1e-6, binding=None)
    projected = (normalized.float() @ down.float().T).bfloat16()
    bottleneck = F.silu(projected[:, :320].float() / 4).bfloat16()
    gates = (bottleneck.float() @ up.float().T).bfloat16()
    expected = Equations.run_gate_mean(normalized, gates, binding=None)
    name = 'input_mix_weight_down_block_inject' if use_combine else 'input_mix_weight_down'
    for rank in range(4):
        monkeypatch.setattr(parallel_state, '_TP', SimpleNamespace(
            rank_in_group=rank, world_size=4))
        monkeypatch.setattr(hc, 'get_tensor_model_parallel_rank', lambda: rank)
        monkeypatch.setattr(linear, 'get_tensor_model_parallel_rank', lambda: rank)
        with torch.device('cuda'), torch.no_grad():
            workspace = hc.HyperConnectionWorkspace(config, 256)
            full = hc.GatedResidual(config, use_combine, workspace=workspace)
            target = getattr(full, name).weight
            target.copy_(down[:target.shape[0]])
            full.input_mix_weight_up.weight.copy_(up)
            full.hc_norm.weight.copy_(norm)
            full._prepare_channel_hc()
            channel = full._channel_hc
            for module in (full, channel):
                module._binding = MethodType(lambda self, x, op: None, module)
                module._plans['combine_norm'] = object()
            calls = []
            def gather(value, dim):
                assert dim == -1
                if not calls:
                    torch.testing.assert_close(value, bottleneck[:, rank*80:(rank+1)*80],
                        atol=.002, rtol=.016)
                    answer = bottleneck
                else:
                    torch.testing.assert_close(value, expected[:, rank*640:(rank+1)*640],
                        atol=.002, rtol=.016)
                    answer = expected
                calls.append(value.shape)
                return answer
            monkeypatch.setattr(hc, 'tensor_model_parallel_all_gather', gather)
            hidden, result, injection = full.combine_and_mix(
                x.clone(), block if pending else None, previous if pending else None)
            torch.testing.assert_close(hidden, expected_hidden, atol=.002, rtol=.016)
            assert len(calls) == (2 if rows <= 128 else 0)
            torch.testing.assert_close(result, expected, atol=.002, rtol=.016)
            if use_combine:
                torch.testing.assert_close(injection, projected[:, 320:324],
                    atol=.002, rtol=.016)
            else:
                assert injection is None
            def forbidden(*args, **kwargs):
                raise AssertionError('Owned rows entered a channel collective')
            monkeypatch.setattr(hc, 'tensor_model_parallel_all_gather', forbidden)
            _, owned, _ = full.combine_and_mix(
                x.clone(), block if pending else None, previous if pending else None,
                replicated_rows=False)
            torch.testing.assert_close(owned, expected, atol=.002, rtol=.016)
            with pytest.raises(ValueError, match='Owned HC rows'):
                channel.combine_and_mix(x, block, previous, replicated_rows=False)
