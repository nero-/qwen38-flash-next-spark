"""Prepare a scoped TP2 port from preserved TP4 reference and image sources."""
from pathlib import Path
import hashlib, json
root = Path(__file__).resolve().parent
ref = root/'tp4-reference'; base = root/'baseline-source'
out = root/'candidate'; tests = root/'tests'
out.mkdir(exist_ok=True); tests.mkdir(exist_ok=True)
def mode(s):
    s=s.replace('from dataclasses import replace', 'from dataclasses import replace\nimport os') if 'from dataclasses import replace' in s else s.replace('from vllm import envs', 'import os\nfrom vllm import envs')
    return s.replace('envs.VLLM_QWEN3_8_HC_PREFILL_MODE', 'os.getenv("VLLM_QWEN3_8_HC_PREFILL_MODE", "off")')
hc_source = mode((ref/'hyperconnection.py').read_text())
hc_source = hc_source.replace('            channel_tp is None\n', '            channel_tp is None\n            and os.getenv("QWEN_HC_CHANNEL_DECODE", "1") != "0"\n', 1)
hc_source = hc_source.replace('config, min(max_tokens, 128), channel_tp=True', 'config, min(max_tokens, max(1, int(os.getenv("QWEN_HC_CHANNEL_DECODE_MAX_TOKENS", "128")))), channel_tp=True')
(out/'hyperconnection.py').write_text(hc_source)
s=mode((ref/'model.py').read_text())
# The TP4 base had a separate recurrent prefill-coalescing change. This port
# only changes HC layout/ownership; preserve this image's GDN implementation.
s=s.replace('                prefill_checkpoint_blocks=int(envs.VLLM_QWEN3_8_PREFILL_COALESCE),\n','')
(out/'model.py').write_text(s)
s=(ref/'hc_prefill.py').read_text().replace('group.world_size != 4','group.world_size not in (2, 4)').replace('BF16 TP4/PP1/DP1/DCP1/PCP1','BF16 TP2-or-TP4/PP1/DP1/DCP1/PCP1')
s=s.replace('    model.hc_prefill_reports = 0','    model.hc_prefill_reports = 0\n    model.hc_tp_size = get_tp_group().world_size if torch.distributed.is_initialized() else 1')
s=s.replace('rows % 4','rows % model.hc_tp_size')
s=s.replace('    group: Any\n','    group: Any\n    tp_size: int\n')
s=s.replace('self.rows // 4','self.rows // self.tp_size')
s=s.replace('RowOwnership(rows, group.rank_in_group, comm)','RowOwnership(rows, group.rank_in_group, comm, group.world_size)')
s=s.replace('rows // 4','rows // model.hc_tp_size')
(out/'hc_prefill.py').write_text(s)
s=(base/'qwen3_next.py').read_text()
old='class Qwen3NextSparseMoeBlock(nn.Module):\n    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):'
new='class Qwen3NextSparseMoeBlock(nn.Module):\n    def __init__(self, vllm_config: VllmConfig, prefix: str = "", *, reduce_results: bool = True):'
assert s.count(old)==1;s=s.replace(old,new)
anchor='        self.experts = FusedMoEFactory(\n'
assert s.count(anchor)==1
s=s.replace(anchor,'''        if not reduce_results and self.replicate_shared_expert:
            raise NotImplementedError("Deferred HC reduction requires partitioned shared experts")

        self.experts = FusedMoEFactory(
            reduce_results=reduce_results,
''')
(out/'qwen3_next.py').write_text(s)
# The unmodified speculator is byte-identical between the two images.
assert (base/'speculator.py').read_bytes() == (ref/'speculator.original.py').read_bytes()
s=(ref/'speculator.py').read_text().replace('SPARKRING_MTP_DRAFT_TOP_K','QWEN_MTP_DRAFT_TOP_K')
(out/'speculator.py').write_text(s)
s=(ref/'test_hybrid_projection.py').read_text()
s=s.replace("@pytest.mark.parametrize('rows'", "@pytest.mark.parametrize('tp_size', [2, 4])\n@pytest.mark.parametrize('rows'")
s=s.replace('def test_channel_partitions_and_owned_rows(monkeypatch, rows, use_combine, pending):','def test_channel_partitions_and_owned_rows(monkeypatch, rows, use_combine, pending, tp_size):')
s=s.replace('lambda: 4','lambda: tp_size').replace('range(4)','range(tp_size)').replace('world_size=4','world_size=tp_size')
s=s.replace('rank*80:(rank+1)*80','rank*(320//tp_size):(rank+1)*(320//tp_size)').replace('rank*640:(rank+1)*640','rank*(2560//tp_size):(rank+1)*(2560//tp_size)')
s=s.replace("@pytest.mark.parametrize('tp_size', [2, 4])", "@pytest.mark.parametrize('channel_decode', [False, True])\n@pytest.mark.parametrize('tp_size', [2, 4])")
s=s.replace('pending, tp_size):', 'pending, tp_size, channel_decode):')
s=s.replace("    monkeypatch.setenv('VLLM_QWEN3_8_FLASH_NEXT_HC_TP', '0')", "    monkeypatch.setenv('QWEN_HC_CHANNEL_DECODE', '1' if channel_decode else '0')\n    monkeypatch.setenv('VLLM_QWEN3_8_FLASH_NEXT_HC_TP', '0')")
s=s.replace('for module in (full, channel):', 'for module in (full, channel) if channel is not None else (full,):')
s=s.replace('(2 if rows <= 128 else 0)', '(2 if channel_decode and rows <= 128 else 0)')
s=s.replace("            with pytest.raises(ValueError, match='Owned HC rows'):\n                channel.combine_and_mix(x, block, previous, replicated_rows=False)", "            if channel is not None:\n                with pytest.raises(ValueError, match='Owned HC rows'):\n                    channel.combine_and_mix(x, block, previous, replicated_rows=False)")
s=s.replace("@pytest.mark.parametrize('channel_decode', [False, True])", "@pytest.mark.parametrize('channel_limit', [4, 128])\n@pytest.mark.parametrize('channel_decode', [False, True])")
s=s.replace('tp_size, channel_decode):', 'tp_size, channel_decode, channel_limit):')
s=s.replace("    monkeypatch.setenv('QWEN_HC_CHANNEL_DECODE',", "    monkeypatch.setenv('QWEN_HC_CHANNEL_DECODE_MAX_TOKENS', str(channel_limit))\n    monkeypatch.setenv('QWEN_HC_CHANNEL_DECODE',")
s=s.replace('channel_decode and rows <= 128', 'channel_decode and rows <= channel_limit')
(tests/'test_hybrid_projection.py').write_text(s)
for name in ['test_draft_topk.py','test_draft_topk_depth.py','test_rejection_sampler_utils.py']:
    text=(ref/name).read_text()
    if name == 'test_draft_topk.py': text=text.replace('151936','248320')
    (tests/name).write_text(text)
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*.py'))}
(root/'candidate-sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
