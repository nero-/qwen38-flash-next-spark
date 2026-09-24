"""Check TP2/TP4 owned-row shapes and reject mixed/captured prefills."""
import importlib.util
import sys
from types import SimpleNamespace as S
import pytest
import torch
spec=importlib.util.spec_from_file_location('candidate_hc_prefill','/tmp/hc_prefill.py')
hc=importlib.util.module_from_spec(spec);sys.modules[spec.name]=hc;spec.loader.exec_module(hc)
@pytest.mark.parametrize('tp',[2,4])
def test_row_partition_and_collective_contract(tp):
    rows=1024; full=torch.arange(rows*8).view(rows,8).float()
    for rank in range(tp):
        count=rows//tp
        class Comm:
            def all_gather(self,out,source):
                torch.testing.assert_close(source,full[rank*count:(rank+1)*count]);out.copy_(full)
            def reduce_scatter(self,out,source):
                torch.testing.assert_close(source,full);out.copy_(full[rank*count:(rank+1)*count]*tp)
        owner=hc.RowOwnership(rows,rank,Comm(),tp)
        local=owner.local(full)
        torch.testing.assert_close(owner.gather(local),full)
        torch.testing.assert_close(owner.reduce(full),local*tp)
        with pytest.raises(ValueError):owner.local(full[:-1])
        with pytest.raises(ValueError):owner.gather(local[:-1])
        with pytest.raises(ValueError):owner.reduce(full[:-1])
@pytest.mark.parametrize('tp',[2,4])
def test_only_pure_eager_prefill_is_eligible(monkeypatch,tp):
    model=S(hc_prefill_mode='shard',hc_tp_size=tp)
    meta=S(num_prefill_tokens=8192,num_prefills=1,num_decodes=0,num_spec_decodes=0)
    ctx=S(is_dummy_run=False,cudagraph_runtime_mode=S(name='NONE'),ubatch_slices=None,attn_metadata={'a':meta})
    monkeypatch.setattr(hc,'is_forward_context_available',lambda:True)
    monkeypatch.setattr(hc,'get_forward_context',lambda:ctx)
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',lambda:False)
    assert hc.eligible(model,8192)
    assert not hc.eligible(model,8191)
    assert not hc.eligible(model,128)
    meta.num_spec_decodes=1;assert not hc.eligible(model,8192);meta.num_spec_decodes=0
    meta.num_decodes=1;assert not hc.eligible(model,8192);meta.num_decodes=0
    meta.num_prefill_tokens=torch.tensor(8192);assert not hc.eligible(model,8192);meta.num_prefill_tokens=8192
    ctx.cudagraph_runtime_mode.name='FULL';assert not hc.eligible(model,8192)
    ctx.cudagraph_runtime_mode.name='NONE';ctx.is_dummy_run=True;assert not hc.eligible(model,8192)
