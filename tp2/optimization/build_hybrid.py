#!/usr/bin/env python3
"""Build the step-5500 hybrid checkpoint: new trunk + main's MXFP8 attention + NVFP4 MTP experts.

Rationale (both model cards): attention and MTP were never distilled. main stores
attention as MXFP8 PTQ and MTP routed experts as W4A16 NVFP4; qad-step5500-ple1000
stores the same source attention in BF16 and MTP experts as MXFP8 PTQ. Swapping
those modules restores main's serving formats while keeping every trained tensor
from step 5500. The MTP head only changes draft acceptance, never target output.

verify  dequantize sample tensors and confirm they come from the same source weights
build   write model-5500h (unchanged shards hardlinked, affected shards rewritten)
Run inside the serving image (torch + safetensors).
"""
import json, os, re, shutil, sys
from collections import defaultdict
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import load_config  # noqa: E402

ROOT = Path.home() / load_config(Path(__file__).resolve().parents[1])["remote_root"]
MAIN, NEW, OUT = ROOT / "model", ROOT / "model-5500", ROOT / "model-5500h"
MTP = ["mtp.layers.0.mlp.experts"]


def load_index(d):
    return json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]


main_map, new_map = load_index(MAIN), load_index(NEW)
main_q = json.loads((MAIN / "hf_quant_config.json").read_text())["quantized_layers"]
new_q = json.loads((NEW / "hf_quant_config.json").read_text())
ATTN = sorted(k for k, v in main_q.items() if v["quant_algo"] == "MXFP8" and k.startswith("model.language_model.layers.")
              and (".linear_attn." in k or ".self_attn." in k))
assert len(ATTN) == 240, len(ATTN)


def members(wmap, prefix):
    return sorted(k for k in wmap if k.startswith(prefix + "."))


def tensor(d, wmap, name):
    with safe_open(str(d / wmap[name]), "pt") as f:
        return f.get_tensor(name)


def dequant_mxfp8(w, s):
    # E4M3 values, UE8M0 scale per 32 contiguous elements along the last dim.
    w = w.to(torch.float32)
    scale = torch.exp2(s.to(torch.float32) - 127.0).repeat_interleave(32, dim=-1)[..., :w.shape[-1]]
    return w * scale


def verify():
    worst = 0.0
    for prefix in ATTN[::24]:
        m = members(main_map, prefix)
        n = members(new_map, prefix)
        assert f"{prefix}.weight" in m and f"{prefix}.weight_scale" in m, m
        assert n == [f"{prefix}.weight"], n
        ref = tensor(NEW, new_map, f"{prefix}.weight").to(torch.float32)
        deq = dequant_mxfp8(tensor(MAIN, main_map, f"{prefix}.weight"), tensor(MAIN, main_map, f"{prefix}.weight_scale"))
        rel = ((deq - ref).norm() / ref.norm()).item()
        cos = torch.nn.functional.cosine_similarity(deq.flatten(), ref.flatten(), dim=0).item()
        worst = max(worst, rel)
        print(json.dumps(dict(tensor=prefix, shape=list(ref.shape), rel_err=round(rel, 5), cosine=round(cos, 6))))
    # MXFP8 relative RMS error is ~2-4%; an unrelated source would be ~100%+.
    assert worst < 0.06, f"attention does not match the same source (worst rel err {worst})"
    for prefix in MTP:
        print(json.dumps(dict(prefix=prefix, main=members(main_map, prefix)[:6], new=members(new_map, prefix)[:6],
                              main_count=len(members(main_map, prefix)), new_count=len(members(new_map, prefix)))))
    other = sorted(k for k in new_map if k.startswith("mtp.") and not any(k.startswith(p + ".") for p in MTP))
    same = sum(torch.equal(tensor(MAIN, main_map, k), tensor(NEW, new_map, k)) for k in other if k in main_map)
    print(json.dumps(dict(mtp_non_expert=len(other), identical_to_main=same, missing_in_main=[k for k in other if k not in main_map][:5])))


def build():
    replace = ATTN + MTP
    drop = {k for p in replace for k in members(new_map, p)}
    add = {k: main_map[k] for p in replace for k in members(main_map, p)}
    assert drop and add and not (set(new_map) - drop) & set(add)
    OUT.mkdir()
    for f in NEW.iterdir():  # small files and untouched shards: hardlink (read-only model trees)
        if f.is_file() and f.name not in ("model.safetensors.index.json", "hf_quant_config.json", "config.json") \
                and not any(new_map[k] == f.name for k in drop):
            os.link(f, OUT / f.name)
    weight_map = {k: v for k, v in new_map.items() if k not in drop}
    by_shard = defaultdict(list)
    for k in drop:
        by_shard[new_map[k]].append(k)
    for shard in sorted(by_shard):  # rewrite affected shards without the replaced tensors
        with safe_open(str(NEW / shard), "pt") as f:
            keep = {k: f.get_tensor(k) for k in f.keys() if k not in drop}
        save_file(keep, str(OUT / shard), metadata={"format": "pt"})
    by_src = defaultdict(list)
    for k, shard in add.items():
        by_src[shard].append(k)
    for i, (shard, keys) in enumerate(sorted(by_src.items()), 1):
        name = f"hybrid-main-{i:05d}.safetensors"
        with safe_open(str(MAIN / shard), "pt") as f:
            save_file({k: f.get_tensor(k) for k in keys}, str(OUT / name), metadata={"format": "pt"})
        weight_map.update({k: name for k in keys})
    index = json.loads((NEW / "model.safetensors.index.json").read_text())
    index["weight_map"] = dict(sorted(weight_map.items()))
    (OUT / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    q = dict(new_q)
    q["quantized_layers"] = dict(new_q["quantized_layers"])
    q["quantized_layers"].update({k: main_q[k] for k in ATTN})
    for k in ("mtp.layers.0.mlp.experts", "mtp.layers.48.mlp.experts"):
        q["quantized_layers"][k] = main_q[k]
    (OUT / "hf_quant_config.json").write_text(json.dumps(q, indent=2))
    cfg = json.loads((NEW / "config.json").read_text())
    cfg["quantization_config"] = q
    (OUT / "config.json").write_text(json.dumps(cfg, indent=2))
    (OUT / "HYBRID.json").write_text(json.dumps(dict(
        trunk="local-inference-lab/Qwen3.8-Flash-Next-NVFP4@60215d26cf5e42c2db6128774032d57fc62678da",
        attention_and_mtp_experts_from="local-inference-lab/Qwen3.8-Flash-Next-NVFP4@7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd",
        replaced_modules=replace, dropped_tensors=len(drop), added_tensors=len(add)), indent=2))
    print(json.dumps(dict(dropped=len(drop), added=len(add), rewritten_shards=len(by_shard), tensors=len(weight_map))))


if __name__ == "__main__":
    {"verify": verify, "build": build}[sys.argv[1]]()
