#!/usr/bin/env python3
"""Score identical token chunks repeatedly on an idle server; report logprob drift.

Healthy batch-invariance noise is ~1e-2 nats per token. Drift of 1e-1 or more,
or multi-nat outliers, indicates nondeterministic numerics in the serving path.
Usage: determinism_probe.py TAG [--lengths 960 1024 2048] [--repeats 3]
"""
import argparse, json, urllib.request
from pathlib import Path

ROOT = Path.home() / "builds/qwen-tp2"
MODEL = "qwen3.8-flash-next-4p89bpw"
p = argparse.ArgumentParser()
p.add_argument("tag")
p.add_argument("--lengths", type=int, nargs="+", default=[960, 1024, 2048])
p.add_argument("--repeats", type=int, default=3)
a = p.parse_args()


def post(path, payload):
    req = urllib.request.Request("http://127.0.0.1:8000" + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


corpus = json.loads((ROOT / "quality-data/corpus-v1.json").read_text())
ids = post("/tokenize", {"model": MODEL, "prompt": corpus["wiki"], "add_special_tokens": False})["tokens"]
out = dict(tag=a.tag, cells=[])
for length in a.lengths:
    chunk = ids[5000:5000 + length]
    runs = []
    for _ in range(a.repeats):
        d = post("/v1/completions", dict(model=MODEL, prompt=chunk, max_tokens=1, temperature=0, echo=True, logprobs=1))
        runs.append(d["choices"][0]["logprobs"]["token_logprobs"][1:length])
    for b in runs[1:]:
        diffs = [abs(x - y) for x, y in zip(runs[0], b)]
        cell = dict(length=length, mean_abs=round(sum(diffs) / len(diffs), 5), max_abs=round(max(diffs), 4),
                    exact=sum(d == 0 for d in diffs), nll=[round(-sum(r) / len(r), 5) for r in (runs[0], b)])
        out["cells"].append(cell)
        print(json.dumps(cell), flush=True)
(ROOT / "evidence" / f"{a.tag}-determinism.json").write_text(json.dumps(out, indent=1))
